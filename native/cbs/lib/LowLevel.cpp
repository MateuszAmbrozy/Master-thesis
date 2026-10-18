#include "LowLevel.h"
#include <cstdint>
#include <queue>
#include <unordered_set>

LowLevel::LowLevel() {}
LowLevel::~LowLevel() {}

bool LowLevel::isBlocked(const Map& map, Position pos, const std::vector<Constraint>&, int) {
    return map.cells[pos.x][pos.y].isObstacle;
}

bool LowLevel::isValid(const Map& map, Position pos, int time, const std::vector<Constraint>& constraints) {
    return pos.x >= 0 && pos.x < static_cast<int>(map.cells.size()) &&
           pos.y >= 0 && pos.y < static_cast<int>(map.cells[0].size()) &&
           !isBlocked(map, pos, constraints, time);
}

bool LowLevel::isConstraint(Constraint constraint, const std::vector<Constraint>& constraints) {
    return std::find(constraints.begin(), constraints.end(), constraint) != constraints.end();
}

bool LowLevel::isDestination(Position pos, Position dest) {
    return pos == dest;
}

double LowLevel::calculateHValue(Position pos, Position dest) {
    return std::abs(pos.x - dest.x) + std::abs(pos.y - dest.y);
}

std::vector<Cell> LowLevel::tracePath(Map&, std::shared_ptr<Cell> cell) {
    std::vector<Cell> path;
    while (cell) {
        path.push_back(*cell);
        cell = cell->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Cell> LowLevel::aStarSearch(Map& map, int agentID, const std::vector<Constraint>& constraints) {
    const Position source = map.agents[agentID].start.position;
    const Position goal = map.agents[agentID].goal.position;
    if (!isValid(map, source, 0, constraints) || !isValid(map, goal, 0, constraints)
        || isConstraint(Constraint(agentID, source, 0), constraints)) {
        return {};
    }
    int latestConstraint = -1;
    int latestGoalConstraint = -1;
    for (const auto& constraint : constraints) {
        if (constraint.agentID != agentID) continue;
        latestConstraint = std::max(latestConstraint, constraint.time);
        if (constraint.position == goal && (!constraint.edge || constraint.from == goal)) {
            latestGoalConstraint = std::max(latestGoalConstraint, constraint.time);
        }
    }
    const int rows = map.cells.size();
    const int columns = map.cells[0].size();
    // After the final constraint, a simple path uses at most |V|-1 moves.
    const int horizon = std::max(1, latestConstraint + rows * columns + 1);
    std::unordered_set<std::uint64_t> closed;
    auto key = [rows, columns](const Cell& cell) {
        return (static_cast<std::uint64_t>(cell.time) * rows + cell.position.x) * columns + cell.position.y;
    };
    std::priority_queue<Cell, std::vector<Cell>, CompareCells> open;
    open.push(Cell(source, 0, calculateHValue(source, goal), calculateHValue(source, goal), 0));
    const int maxExpansions = 300000;
    int expansions = 0;
    const int directions[5][2] = {{0,0}, {0,1}, {1,0}, {0,-1}, {-1,0}};
    while (!open.empty()) {
        Cell current = open.top();
        open.pop();
        if (!closed.insert(key(current)).second) continue;
        if (++expansions > maxExpansions) return {};
        // Ending a path implies staying on the goal at every later timestep.
        if (current.position == goal && current.time > latestGoalConstraint) {
            return tracePath(map, std::make_shared<Cell>(current));
        }
        if (current.time >= horizon) continue;
        for (const auto& direction : directions) {
            Position next(current.position.x + direction[0], current.position.y + direction[1]);
            const int arrival = current.time + 1;
            if (!isValid(map, next, arrival, constraints)
                || isConstraint(Constraint(agentID, next, arrival), constraints)
                || isConstraint(Constraint(agentID, current.position, next, arrival), constraints)) {
                continue;
            }
            const double g = current.g + 1;
            const double h = calculateHValue(next, goal);
            Cell candidate(next, g, h, g+h, arrival, std::make_shared<Cell>(current));
            if (closed.count(key(candidate)) == 0) open.push(candidate);
        }
    }
    return {};
}

std::vector<std::vector<Cell>> LowLevel::getPaths(const std::vector<Constraint>& constraints, Map& map) {
    std::vector<std::vector<Cell>> paths;
    for (int agent = 0; agent < static_cast<int>(map.agents.size()); ++agent) {
        auto path = aStarSearch(map, agent, constraints);
        if (path.empty()) return {};
        paths.push_back(std::move(path));
    }
    return paths;
}
