#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCULENN.h"
#else

TH_API void THLENN_(LenSoftMax_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *len);

TH_API void THLENN_(LenSoftMax_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  THCIndexTensor *len);

#endif
