#include <THC/THC.h>

#define THCIndexTensor THCudaLongTensor
#define THCIndexTensor_(NAME) THCudaLongTensor_ ## NAME
typedef long THCIndex_t;

#define THLENN_(NAME) TH_CONCAT_3(THLENN_, CReal, NAME)

#include "generic/THCULENN.h"
#include <THC/THCGenerateFloatTypes.h>
