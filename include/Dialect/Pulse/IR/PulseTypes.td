//===- PulseTypes.td - Pulse dialect types -----------------*- tablegen -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef PULSE_TYPES
#define PULSE_TYPES

include "mlir/IR/OpBase.td"

class Pulse_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PulseDialect, name, traits> {
        let mnemonic = typeMnemonic;
}

def Pulse_CaptureType : Pulse_Type<"Capture", "capture"> {
    let summary = "TODO summary.";
    let description = [{
        TODO description. (I have no idea what this is supposed to be.)
    }];
}

// TODO: The roles of Port and Frame must be disambiguated and deduplicated in many places.
def Pulse_FrameType : Pulse_Type<"Frame", "frame"> {
    let summary = "Frame of reference for interacting with qubits.";
    let description = [{
        The frame type is an abstraction for a frame of reference required for interacting with qubits.
    }];
}

def Pulse_KernelType : Pulse_Type<"Kernel", "kernel"> {
    let summary = "An integration kernel used for capturing.";
    let description = [{
        A kernel is an abstraction that defines steps for integrating and discrimination of a measurement.
    }];
}

// TODO: The roles of Port and Frame must be disambiguated and deduplicated in many places.
def Pulse_PortType : Pulse_Type<"Port", "port"> {
    let summary = "An input or output for manipulating and observing qubits.";
    let description = [{
        A port is an abstraction of an input or output that is intended to interact with qubits.
    }];
}

// TODO: The requirement for PortGroup is unclear. It is likely possible to remove its existence.
def Pulse_PortGroupType : Pulse_Type<"PortGroup", "port_group"> {
    let summary = "A collection of ports.";
    let description = [{
        To be replaced or removed.
    }];
}

def Pulse_WaveformType : Pulse_Type<"Waveform", "waveform"> {
    let summary = "A waveform for interacting with qubits.";
    let description = [{
        The waveform type represents a waveform (or waveform envelope) used for interacting with qubits.
    }];
}

def Pulse_MixedFrameType : Pulse_Type<"MixedFrame", "mixed_frame"> {
    let summary = "A frame associated with a port for interacting with qubits.";
    let description = [{
        The mixed frame type represents a frame associated with a port used for interacting with qubits.
    }];
}

def AnyFrame : AnyTypeOf<[Pulse_FrameType, Pulse_MixedFrameType]>;

def AnyPulse : AnyTypeOf<[Pulse_CaptureType, Pulse_FrameType, Pulse_KernelType, Pulse_PortType,
                          Pulse_PortGroupType, Pulse_WaveformType, Pulse_MixedFrameType]>;

#endif // PULSE_TYPES
