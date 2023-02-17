@0xdf236ae67a45a070;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("qssc::payload::schema");

struct QuantumExecutionModule {
    components @0: List(Component);
}

struct Component {
    # Represents a Mock{Controller, Instrument, Acquire, Drive}

    uid @0: Text;       # component unique ID
    config @1: Data;    # configuration data
    program @2: Data;   # controller binary, .mlir, .ll
}
