from ngclearn import Context
from .component_to_lava import to_lava as component_to_lava
from lava.magma.core.process.variable import Var
from ngcsimlib.logger import info
from ngclearn import Compartment


class LavaContext(Context):
    def __init__(self, name):
        super().__init__(name)
        self._rebuild_lava = True

        if hasattr(self, "_init_lava"):
            return

        self._init_lava = True

        self.dynamic_lava_processes = {}
        self.dynamic_lava_models = {}
        self.mapped_processes = {}
        self.lagging_components = {}

    @property
    def updater(self):
        self._rebuild_lava = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._rebuild_lava:
            self._rebuild_lava = False
            info("Rebuilding lava components")
            self._update_dynamic_class()
            self._build_lava_processes()
            self._wire_lava_processes()


        super().__exit__(exc_type, exc_val, exc_tb)

    def set_lag(self, component_name, status):
        self.lagging_components[component_name] = status

    def get_lava_components(self, *args):
        _components = []
        for a in args:
            if a in self.mapped_processes.keys():
                _components.append(self.mapped_processes[a])
        return _components

    def write_to_ngc(self):
        for p_name, lc in self.mapped_processes.items():
            for a_name, a in lc.__dict__.items():
                if isinstance(a, Var):
                    if Compartment.is_compartment(self.components[p_name].__dict__[a_name]):
                        self.components[p_name].__dict__[a_name].set(a.get())


    def _update_dynamic_class(self):
        info("updating dynamic classes")
        for k, v in self.components.items():
            process, model = component_to_lava(v, lag=self.lagging_components.get(k, False))
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

