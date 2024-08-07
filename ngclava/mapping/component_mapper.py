"""
Contains the method for mapping a ngclearn component to a lava component.
This is mostly used as a helper method and should not be called by itself.

This method does muddle the globals dictionary with uuids as they are needed
for lava to bind processes to models.
"""
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from ngcsimlib.compilers.component_compiler import parse
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.ports.ports import InPort, OutPort
from ngclearn import numpy as np

import uuid

def map_component(source_obj, lag=False):
    """
    Dynamically makes a lava process and a lava model class based off the source
    object provided.

    Args:
        source_obj: The source object to model after

        lag: is this is a lagging component (See lava context.set_lag() for more
        details)

    Returns: dynamic_process, dynamic_model

    """
    (pure_fn, output_compartments, args, parameters, compartments) = parse(source_obj, "advance_state")
    args = []
    assert len(args) == 0
    all_vals = {**{p: source_obj.__dict__[p] for p in parameters},
                **{c: source_obj.__dict__[c].value for c in compartments},
                **{oc: source_obj.__dict__[oc].value for oc in output_compartments}}

    try:
        (pure_reset, output_compartments_reset, args_reset, parameters_reset, compartments_reset) = parse(source_obj, "reset")
    except:
        (pure_reset, output_compartments_reset, args_reset, parameters_reset, compartments_reset) = None, None, None, None, None



    class dynamic_lava_process(AbstractProcess):
        def __init__(self, source_object, **kwargs):
            super().__init__(**kwargs)

            for k, v in all_vals.items():
                val = source_object.__dict__.get(k, kwargs.get(k, 0))
                val = val.value if hasattr(val, "value") else val
                self.__dict__[k] = Var(v.shape if hasattr(v, 'shape') else (1,), val, name=k)

            for conn in source_obj.connections:
                c_name = conn.destination.name.split("/")[-1]
                if c_name in all_vals.keys():
                    self.__dict__["_inp_" + c_name] = InPort(shape=self.__dict__[c_name].shape)

            for oc in output_compartments:
                self.__dict__["_out_" + oc] = OutPort(shape=self.__dict__[oc].shape)

        def reset(self):
            if pure_reset is None:
                return

            funParams = {narg: source_obj.__dict__[narg] for narg in list(parameters_reset)}
            funComps = {narg: self.__dict__[narg] for narg in list(compartments_reset)}

            vals = pure_reset.__func__(**funParams, **funComps)
            for key, v in zip(output_compartments_reset, vals):
                # self.__dict__[key] = Var(v.shape if hasattr(v, 'shape') else (1,), v, name=key)
                self.__dict__[key].set(v)

    @implements(proc=dynamic_lava_process, protocol=LoihiProtocol)
    @requires(CPU)
    @tag('floating_pt')
    class dynamic_lava_model(PyLoihiProcessModel):
        def run_spk(self):
            if lag:
                #T-1 Outputs
                for oc in output_compartments:
                    self.__dict__["_out_" + oc].send(np.reshape(self.__dict__[oc], source_obj.__dict__[oc].value.shape))

            #Gather
            for comp in compartments:
                if hasattr(self, "_inp_" + comp):
                    self.__dict__[comp] = self.__dict__["_inp_" + comp].recv()

            #Run Dynamics
            _param_loc = 0
            _comps_loc = 0

            funParams = {narg: self.__dict__[narg] for narg in list(parameters)}
            funComps = {narg: self.__dict__[narg] for narg in list(compartments)}

            vals = pure_fn.__func__(**funParams, **funComps)
            if len(output_compartments) == 1:
                vals = [vals]

            for key, v in zip(output_compartments, vals):
                self.__dict__[key] = v

            if not lag:
                #Output
                for oc in output_compartments:
                    self.__dict__["_out_" + oc].send(np.reshape(self.__dict__[oc], source_obj.__dict__[oc].value.shape))



    #Make Inputs
    for conn in source_obj.connections:
        c_name = conn.destination.name.split("/")[-1]
        if c_name in all_vals.keys():
            inp_name = "_inp_" + conn.destination.name.split("/")[-1]
            setattr(dynamic_lava_model, inp_name, LavaPyType(PyInPort.VEC_DENSE, float))

    #Make Outputs
    for oc in output_compartments:
        oc_name = "_out_" + oc
        setattr(dynamic_lava_model, oc_name, LavaPyType(PyOutPort.VEC_DENSE, float, precision=1))

    for k, v in all_vals.items():
        setattr(dynamic_lava_model, k, LavaPyType(v.__class__, float))

    globals()[str(uuid.uuid4())] = dynamic_lava_model

    return dynamic_lava_process, dynamic_lava_model


