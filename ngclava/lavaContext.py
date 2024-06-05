"""
This is an extension of the base ngclearn context. This context automatically
builds a mirrored model using lava compatible processes and models.
"""

from ngclearn import Context
from ngclava.mapping.component_mapper import map_component
from lava.magma.core.process.variable import Var
from ngcsimlib.logger import info, warn
from ngclearn import Compartment
from ngclava import lava_compatible_env


class LavaContext(Context):
    """
    The lava context is built on top of the default ngclearn context and thus has
    all the same functionality for model building and model training that the base
    class has. What the lava context adds to this is a mirrored lava model that can
    be run on an Intel Loihi-2 chip with minimal added overhead. There are some
    restrictions when building lava contexts defined below

    - Only components found in ngclearn/components/lava are guaranteed to be
    compatible with lava

    - Components can not make use of JAX specific methods that are not found in base numpy

    - All random initializing must be done with base numpy

    - The flag "packages/use_base_numpy" in the configuration file for ngclearn must be set to True

    - When writing custom components for lava import numpy from ngclearn as this will default to JAX
    for GPU training but will swap to the base package numpy if `packages/use_base_numpy` is set to true

    There are only certain times when the context will rebuild all the lava components.
    rebuilding of the components is needed any time there is a change in structure or parameters
    to the model as lava requires a static topology (similar to compiled command in ngclearn).
    The times that the context will automatically rebuild the lava components is when
    - it exits the initial constructor
    - is call via `with model.updater:`
    - it exits from a constructor call that returned an existing model

    The times that the context will not automatically rebuild the lava components
    is when `with model:` is called.

    A manual rebuild can be triggered at any point with `model.rebuild_lava()`

    It is important to know that all dynamic's defined commands that referenced
    lava components will no longer work post rebuild as all previous lava components
    are no longer used (They will still technically exist so no errors will be thrown).

    """

    def __init__(self, name):
        super().__init__(name)
        self._rebuild_lava = lava_compatible_env()

        if hasattr(self, "_init_lava"):
            return

        self._init_lava = True

        self.dynamic_lava_processes = {}
        self.dynamic_lava_models = {}
        self.mapped_processes = {}
        self.lagging_components = {}

    @property
    def updater(self):
        """
        Simply a property to be called with `with lavaContext.updater` to trigger
        a rebuild of the lava components upon exiting the with block.
        Will not rebuild if the current import environment is not compatible with lava
        """
        self._rebuild_lava = lava_compatible_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._rebuild_lava:
            self._rebuild_lava = False
            self.rebuild_lava()
        super().__exit__(exc_type, exc_val, exc_tb)

    def rebuild_lava(self):
        """
        Triggers a manual rebuild of the lava components
        """
        if not lava_compatible_env():
            warn("The current environment is not compatible to build lava objects")
            return
        info("Rebuilding lava components")
        self._update_dynamic_class()
        self._build_lava_processes()
        self._wire_lava_processes()

    def set_lag(self, component, status=True):
        """
        In lava, it is easy to lock your system if there is recurrence in your model.
        The lava context allows for you to lag the values emitted by specific components
        to be from the previous timestep.

        By default, the process pattern for a mapped lava component is
        Receive values -> Process values -> Emit values

        A lagged lava component will follow the pattern
        Emit values -> Receive values -> Process Values

        Example:
            There is a model that has the wiring pattern of `Z0 -> W1 -> Z1 -> W1`
            Here we can see that in order for Z1 to emit values it relies on the values
            emitted by W1. But W1 also relies on values emitted from Z1. So if we lag
            W1 it will emit last timesteps value at the start of the loop and then wait
            for the new values meaning that the value emitted by W1 will be delayed by a
            timestep, but it will no longer lock Z1 from running.

        Args:
            component: Either the component to lag, or the name of the component to lag

            status: a boolean for if the component should or should not be
                lagged (default: True)
        """
        if isinstance(component, str):
            self.lagging_components[component] = status
        else:
            self.lagging_components[component.name] = status

    def get_lava_components(self, *component_names, unwrap=True):
        """
        Mirrors ngclearn.Context.get_components()
        Gets all the components by name in a context

        Args:
            component_names: an arbitrary list of component names to get

            unwrap: return just the component not a list of length 1 if only a single component is retrieved

        Returns:
             either a list of components or a single component depending on the number of components being retrieved
        """
        if len(component_names) == 0:
            return None
        _components = []
        for a in component_names:
            if a in self.mapped_processes.keys():
                _components.append(self.mapped_processes[a])
            else:
                warn(f"Could not fine a lava component with the name \"{a}\" in the context")
        return _components if len(component_names) > 1 or not unwrap else _components[0]


    def write_to_ngc(self):
        """
        Copies all the current values of the lava model into the ngc model
        """
        for p_name, lc in self.mapped_processes.items():
            for a_name, a in lc.__dict__.items():
                if isinstance(a, Var):
                    if Compartment.is_compartment(self.components[p_name].__dict__[a_name]):
                        self.components[p_name].__dict__[a_name].set(a.get())

    def _update_dynamic_class(self):
        info("updating dynamic classes")
        for k, v in self.components.items():
            process, model = map_component(v, lag=self.lagging_components.get(k, False))
            self.dynamic_lava_processes[k] = process
            self.dynamic_lava_models[k] = model

    def _build_lava_processes(self):
        info("building lava processes")
        for k, v in self.components.items():
            self.mapped_processes[k] = self.dynamic_lava_processes[k](v, name=k)

    def _wire_lava_processes(self):
        info("wiring lava processes")
        for k, v in self.components.items():
            for conn in v.connections:

                dest_component, dest_compartment = conn.destination.name.split("/")
                if dest_compartment not in self.mapped_processes[dest_component].__dict__.keys():
                    continue

                dest = self.mapped_processes[dest_component].__dict__["_inp_" + dest_compartment]
                for source in conn.sources:
                    source_component, source_compartment = source.name.split("/")
                    sor = self.mapped_processes[source_component].__dict__["_out_" + source_compartment]
                    dest.connect_from(sor)
