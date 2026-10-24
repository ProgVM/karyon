#include <iostream>
#include <vector>
#include <cmath>

int main() {
    float phi = 0.0f;
    float pi = 0.0f;
    float dt = 0.005f;
    float K = 0.5f;
    float M = -0.2f;
    float N = -0.15f;
    
    std::cout << "Testing 1D oscillator with cubic..." << std::endl;
    for (int t = 0; t < 100; ++t) {
        float j_ext = 0.1f * sinf(t * 0.1f);
        float accel = M * phi - 0.15f * phi * phi * phi + N * pi + j_ext;
        pi += accel * dt;
        phi += pi * dt;
        if (t % 10 == 0) {
            std::cout << "t=" << t << " phi=" << phi << " pi=" << pi << " accel=" << accel << std::endl;
        }
    }
    return 0;
}
