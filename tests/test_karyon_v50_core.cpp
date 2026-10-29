/* Sanity Test for Karyon v50.0 Master Shared Library */
#include <iostream>
#include <vector>
#include "karyon_core.h"

int main() {
    std::cout << "[Sanity Test] Initializing Karyon v50.0 Shared Core..." << std::endl;
    karyon::SovereignFullPlasticityOrganism organism;
    
    std::string test_data = "Karyon-CoRE Sovereign Autopoietic Multi-Scale Core Test\n";
    for (size_t t = 0; t < test_data.length() - 1; ++t) {
        uint8_t in_byte = (uint8_t)test_data[t];
        uint8_t targ_byte = (uint8_t)test_data[t + 1];
        uint8_t decision = organism.perceive_and_decide(in_byte);
        organism.assimilate_experience(in_byte, 1.0f, 1.0f, targ_byte);
    }
    
    auto telem = organism.get_telemetry();
    std::cout << "  Active Dim   : " << telem.active_dim << " / 256" << std::endl;
    std::cout << "  Active Layers: " << telem.active_layers << " / 6" << std::endl;
    std::cout << "  Free Energy  : " << telem.free_energy << std::endl;
    std::cout << "[Sanity Test] Karyon v50.0 Master Core Initialization & Execution SUCCESSFUL!" << std::endl;
    return 0;
}
