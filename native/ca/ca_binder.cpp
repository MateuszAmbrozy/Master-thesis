#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <vector>
#include <utility>
#include <stdexcept>


#include "lib/CA_Environment.h"
#include "lib/Graph.h"
#include "lib/Agent.h"

namespace py = pybind11;


std::vector<std::vector<std::pair<int, int>>> find_path_wrapper(
    std::vector<std::pair<int, int>> obstacles,
    std::vector<std::pair<int, int>> starts,
    std::vector<std::pair<int, int>> goals,
    double timeout = 10.0) 
{
    if (starts.size() != goals.size()) {
        throw std::invalid_argument("The number of starts must equal the number of goals.");
    }

    
    int max_x = 0, max_y = 0;
    auto update_dims = [&](const std::pair<int, int>& p) {
        if (p.first > max_x) max_x = p.first;
        if (p.second > max_y) max_y = p.second;
    };
    
    for (const auto& p : obstacles) update_dims(p);
    for (const auto& p : starts) update_dims(p);
    for (const auto& p : goals) update_dims(p);

    int width = max_x + 1;
    int height = max_y + 1;

    
    std::vector<map::Cell> obs_cells;
    for (const auto& obs : obstacles) {
        obs_cells.push_back(map::Cell(obs.first, obs.second, true));
    }

    
    map::Graph graph(width, height, obs_cells);

    
    std::vector<Agent> agents;
    for (size_t i = 0; i < starts.size(); ++i) {
        map::Cell startPos(starts[i].first, starts[i].second);
        map::Cell goalPos(goals[i].first, goals[i].second);
        
        agents.emplace_back(i, startPos, goalPos);
    }

    
    CA_Environment env(agents, graph);
    
    
    env.planAllPaths(); 

    
    std::vector<std::vector<std::pair<int, int>>> python_result;
    
    for (Agent agent : env.getAgents()) {
        std::vector<CA::Cell> path = agent.getPath();
        std::vector<std::pair<int, int>> py_path;
        
        for (const auto& cell : path) {
            py_path.push_back({cell.x, cell.y});
        }
        
        
        
        python_result.push_back(py_path);
    }

    return python_result;
}



PYBIND11_MODULE(ca_core, m) {
    m.doc() = "Cooperative A* (CA) implemented in C++";

    m.def("find_path", &find_path_wrapper, "Find collision-free multi-agent paths in space and time.",
          py::arg("obstacles"), 
          py::arg("starts"), 
          py::arg("goals"), 
          py::arg("timeout") = 10.0);
}