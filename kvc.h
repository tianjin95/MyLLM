#pragma once

#include "llm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

class KVC {

    public:
    // Keep one K/V pair per transformer layer. The runtime should own one KVC
    // instance for each layer rather than sharing this pair globally.
    Matrix kc;
    Matrix vc;

    KVC() = default;
    
};

}
