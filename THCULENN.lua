local ffi = require 'ffi'
local THLENN = require 'lenn.THLENN'

local THCULENN = {}

-- load libTHCULENN
THCULENN.C = ffi.load(package.searchpath('libTHCULENN', package.cpath))

-- load THC
local THC = ffi.os == 'Windows' and ffi.load('THC') or ffi.C

local THCState_ptr = ffi.typeof('THCState*')

function THCULENN.getState()
   return THCState_ptr(cutorch.getState());
end

local THCULENN_generic_h = require 'culenn.THCULENN_generic_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THLENN.h
THCULENN_generic_h = THCULENN_generic_h:gsub("\n#[^\n]*", "")
THCULENN_generic_h = THCULENN_generic_h:gsub("^#[^\n]*\n", "")

local preprocessed_generic = string.gsub(THCULENN_generic_h, 'TH_API void THLENN_%(([%a%d_]+)%)', 'void THLENN_TYPE%1')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THCIndexTensor'] = 'THCudaLongTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'float'
   }
}

local cct2lt = {
   ['THCudaFloatTensor'] = 'torch.CudaTensor',
   ['THCudaDoubleTensor'] = 'torch.CudaDoubleTensor',
}

local replacements_generic =
{
  {
    ['THCTensor'] = 'THCudaTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'Cuda',
    ['accreal'] = 'float',
  },
  {
    ['THCTensor'] = 'THCudaDoubleTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaDouble',
    ['accreal'] = 'double',
   }
}

if cutorch.hasHalf then
  ffi.cdef("half THC_float2half(float a);")
  ffi.cdef("float THC_half2float(half a);")
  cct2lt['THCudaHalfTensor'] = 'torch.CudaHalfTensor'
  local half_replacement = {
    ['THCTensor'] = 'THCudaHalfTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaHalf',
    ['accreal'] = 'float',
  }
  table.insert(replacements_generic, half_replacement)
end

for i=1,#replacements_generic do
    local r = replacements_generic[i]
    local s = preprocessed_generic
    for k,v in pairs(r) do
        s = string.gsub(s, k, v)
    end
    ffi.cdef(s)
end

local function extract_function_names_generic(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THLENN_%(([%a%d_]+)%)') do
       t[#t+1] = n
   end
   return t
end

local function find_positions(s, p)
   local begin = 0
   local positions = {}
   while true do
      local start, stop = string.find(s, p, begin)
      if (start == nil) then break end
      positions[#positions+1] = start
      begin = stop + 1
   end
   return positions
end

local function extract_function_names_and_real_args(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API ([^;]+)') do
      local func_name = string.match(n, 'void THLENN_%(([%a%d_]+)%)')
      local param_positions = find_positions(n, ',')
      local positions = {}
      for x,y in ipairs(find_positions(n, 'real')) do
          local found = false
          for cn,cp in ipairs(param_positions) do
              if cp > y then
                positions[#positions+1] = cn
                found = true
                break
              end
          end
          -- it is the last param
          if not found then positions[#positions+1] = #param_positions + 1 end
      end

   t[func_name] = positions
   end
   return t
end

local real_args = extract_function_names_and_real_args(THCULENN_generic_h)

-- build function table
local function_names_generic = extract_function_names_generic(THCULENN_generic_h)

THLENN.kernels['torch.CudaTensor'] = THLENN.bind(THCULENN.C, function_names_generic, 'Cuda', THCULENN.getState)
torch.getmetatable('torch.CudaTensor').THLENN = THLENN.kernels['torch.CudaTensor']

THLENN.kernels['torch.CudaDoubleTensor'] = THLENN.bind(THCULENN.C, function_names_generic, 'CudaDouble', THCULENN.getState)
torch.getmetatable('torch.CudaDoubleTensor').THLENN = THLENN.kernels['torch.CudaDoubleTensor']

if cutorch.hasHalf then
   local raw_half_functions = THLENN.bind(THCULENN.C, function_names_generic, 'CudaHalf', THCULENN.getState)
   THLENN.kernels['torch.CudaHalfTensor'] = raw_half_functions
   torch.getmetatable('torch.CudaHalfTensor').THLENN = THLENN.kernels['torch.CudaHalfTensor']
end

local function Module__converter(type)
    return function(self)
            return self:type(type)
    end
end

rawset(torch.getmetatable('nn.Module'), 'cudaDouble', Module__converter('torch.CudaDoubleTensor'))
if cutorch.hasHalf then
    rawset(torch.getmetatable('nn.Module'), 'cudaHalf', Module__converter('torch.CudaHalfTensor'))
end
return THCULENN
