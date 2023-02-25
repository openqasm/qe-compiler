
#include "Payload/Payload.h"

namespace qssc::payload::mock {

// Register the mock target.
int init();

class MockPayload : public Payload {
public:
  MockPayload();
  virtual ~MockPayload() = default;

  // write all files to the stream
  void write(llvm::raw_ostream &stream) final;
  // write all files to the stream
  void write(std::ostream &stream) final;
  // write all files in plaintext to the stream
  void writePlain(std::ostream &stream) final;
  void writePlain(llvm::raw_ostream &stream) final;
};

} // namespace qssc::payload::mock