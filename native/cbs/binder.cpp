#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "lib/HighLevel.h"
#include "lib/util.h"
#include <stdexcept>
#include <vector>
#include <utility>

namespace py = pybind11;


std::vector<std::vector<std::pair<int, int>>> find_path_wrapper(
    std::vector<std::pair<int, int>> obstacles,
    std::vector<std::pair<int, int>> starts,
    std::vector<std::pair<int, int>> goals,
    double timeout = 5.0) 
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

    
    Map map;
    map.cells.resize(width, std::vector<Cell>(height, Cell(Position(0, 0))));
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            map.cells[i][j] = Cell(Position(i, j));
        }
    }

    
    for (const auto& obs : obstacles) {
        map.cells[obs.first][obs.second].isObstacle = true;
    }

    
    for (size_t i = 0; i < starts.size(); ++i) {
        Position startPos(starts[i].first, starts[i].second);
        Position goalPos(goals[i].first, goals[i].second);
        
        map.agents.push_back(Agent(i + 1, startPos, goalPos)); 
    }

    
    std::vector<Constraint> initial_constraints;
    HighLevel hl(map, initial_constraints);
    
    
    
    
    auto raw_result = hl.solve(timeout);

    
    
    std::vector<std::vector<std::pair<int, int>>> python_result;
    
    for (const auto& path : raw_result) {
        std::vector<std::pair<int, int>> py_path;
        for (const auto& cell : path) {
            py_path.push_back({cell.position.x, cell.position.y});
        }
        python_result.push_back(py_path);
    }

    return python_result;
}



PYBIND11_MODULE(cbs_core, m) {
    m.doc() = "CBS (Conflict-Based Search) implemented in C++";

    m.def("find_path", &find_path_wrapper, "Find collision-free multi-agent paths.",
          py::arg("obstacles"), 
          py::arg("starts"), 
          py::arg("goals"), 
          py::arg("timeout") = 10.0);
}