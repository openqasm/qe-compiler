#include "API/api.h"

int main(int argc, const char **argv) {
  return qssc::bind(argc, argv, nullptr, {});
}