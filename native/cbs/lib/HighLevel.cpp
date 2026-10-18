#include "HighLevel.h"
#include <algorithm>
#include <chrono>

HighLevel::HighLevel(Map &map, const std::vector<Constraint>& constraints)
{
    this->map = map;
    CT = std::make_unique<ConstraintTree>(map);
    root->constraints = constraints;
    root->solution = ConstraintTree::findInitialSolution(constraints);
    root->updateCost();
    if (root->solution.size() == map.agents.size()) {
        openList.push(root);
    }
}

std::vector<std::vector<Cell>> HighLevel::solve(double timeout)
{
    const auto started = std::chrono::steady_clock::now();
    while (!openList.empty()) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() > timeout) {
            return {};
        }
        auto node = openList.top();
        openList.pop();
        const bool vertex = ConstraintTree::detectConflicts(*node);
        if (!vertex && !ConstraintTree::hasEdgeConflict(*node)) {
            return node->solution;
        }
        const Conflict conflict = ConstraintTree::getFirstConflict(*node);
        for (int branch = 0; branch < 2; ++branch) {
            const int agent = branch == 0 ? conflict.agent1 : conflict.agent2;
            Constraint constraint(agent, conflict.pos1.position, conflict.time);
            if (!vertex) {
                // pos1/pos2 are the agents' destinations after their swapped move.
                const Position from = branch == 0 ? conflict.pos2.position : conflict.pos1.position;
                const Position to = branch == 0 ? conflict.pos1.position : conflict.pos2.position;
                constraint = Constraint(agent, from, to, conflict.time);
            }
            if (std::find(node->constraints.begin(), node->constraints.end(), constraint) != node->constraints.end()) {
                continue;
            }
            auto child = node->addConstraint(constraint);
            child->updateSolution(map, node->solution, node->cost);
            child->updateCost();
            if (child->solution.size() == map.agents.size() && child->cost < DBL_MAX) {
                openList.push(child);
            }
        }
    }
    return {};
}
