//===- PulseTypes.td - Pulse dialect types -----------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
    let summary = "a frame of reference for interacting with qubits";
    let description = [{
        The frame type is an abstraction for a frame of reference required for interacting with qubits.
    }];
}

def Pulse_KernelType : Pulse_Type<"Kernel", "kernel"> {
    let summary = "an integration kernel used for capturing";
    let description = [{
        A kernel is an abstraction that defines steps for integrating and discrimination of a measurement.
    }];
}

// TODO: The roles of Port and Frame must be disambiguated and deduplicated in many places.
def Pulse_PortType : Pulse_Type<"Port", "port"> {
    let summary = "an input or output for manipulating and observing qubits";
    let description = [{
        A port is an abstraction of an input or output that is intended to interact with qubits.
    }];
}

// TODO: The requirement for PortGroup is unclear. It is likely possible to remove its existence.
def Pulse_PortGroupType : Pulse_Type<"PortGroup", "port_group"> {
    let summary = "a collection of ports";
    let description = [{
        To be replaced or removed.
    }];
}

def Pulse_WaveformType : Pulse_Type<"Waveform", "waveform"> {
    let summary = "a waveform for interacting with qubits";
    let description = [{
        The waveform type represents a waveform (or waveform envelope) used for interacting with qubits.
    }];
}

def Pulse_MixedFrameType : Pulse_Type<"MixedFrame", "mixed_frame"> {
    let summary = "a frame associated with a port for interacting with qubits";
    let description = [{
        The mixed frame type represents a frame associated with a port used for interacting with qubits.
    }];
}

def AnyFrame : AnyTypeOf<[Pulse_FrameType, Pulse_MixedFrameType]>;

def AnyPulse : AnyTypeOf<[Pulse_CaptureType, Pulse_FrameType, Pulse_KernelType, Pulse_PortType,
                          Pulse_PortGroupType, Pulse_WaveformType, Pulse_MixedFrameType]>;

#endif // PULSE_TYPES
