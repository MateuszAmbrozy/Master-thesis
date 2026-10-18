#pragma once
#include "util.h"
#include "LowLevel.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <queue>

class TreeNode: public std::enable_shared_from_this<TreeNode> {
public:
    std::vector<Constraint> constraints;
    std::vector<std::vector<Cell>> solution; 
    double cost;
    std::shared_ptr<TreeNode> parent;

    TreeNode(std::shared_ptr<TreeNode> parent = nullptr)
        : cost(DBL_MAX), parent(parent) {}

    // Function to add a constraint and create a new node
    std::shared_ptr<TreeNode> addConstraint(const Constraint& constraint) {
        auto child = std::make_shared<TreeNode>(shared_from_this());
        child->constraints = this->constraints;
        child->constraints.push_back(constraint);
        return child;
    }

    void updateSolution(Map& map,const std::vector<std::vector<Cell>>& solution, double cost) {
        LowLevel solver;
        this->solution = solver.getPaths(constraints, map);
        this->cost = cost;
    }
    void updateCost()
    {
        if (solution.empty()) {
            this->cost = DBL_MAX;
            return;
        }
        auto cost = 0;
        for (auto route : solution) {
            cost += route.size() - 1;
        }
        this->cost = cost;
    }
};

struct CompareNodes //comaprition to priority_queue
{
    bool operator()(const std::shared_ptr<TreeNode>& a, const std::shared_ptr<TreeNode>& b) const {
        return a->cost > b->cost;
    }
};

class ConstraintTree {


private:
    static Map map;
public:
    ConstraintTree(Map& map)
        {
            this->map = map;
        }

    static std::vector<std::vector<Cell>> findInitialSolution(const std::vector<Constraint>& constraints) {
        LowLevel solver;
        return solver.getPaths(constraints, map);
    }
    static bool detectConflicts(const std::vector<Cell> &route1, const std::vector<Cell> &route2) {
        size_t max_index = std::max(route1.size(), route2.size());
        for (size_t k = 0; k < max_index; k++) {
            Cell pos1 = (k < route1.size()) ? route1[k] : route1.back();
            Cell pos2 = (k < route2.size()) ? route2[k] : route2.back();

            if (pos1 == pos2) {
                return true;
            }
        }
        return false;
    }
    static bool detectConflicts( const TreeNode& node) {
        auto solutions = node.solution;
        for (int i = 0; i < solutions.size(); i++) {
            for (int j = i + 1; j < solutions.size(); j++) {
                if (detectConflicts(solutions[i], solutions[j]))
                    return true;
            }
        }
        return false;
    }

    static bool hasEdgeConflict(const std::vector<Cell> &route1, const std::vector<Cell> &route2) {
        size_t min_route_size = std::min(route1.size(), route2.size());
            if (min_route_size < 2) return false;

            for (size_t k = 0; k < min_route_size - 1; k++) {
                if (route1[k] == route2[k + 1] && route1[k + 1] == route2[k])
                    return true;
            }
            return false;
    }

    static bool hasEdgeConflict(const TreeNode &node) {
        auto solutions = node.solution;

        for (int i = 0; i < solutions.size(); i++) {
            for (int j = i + 1; j < solutions.size(); j++) {
                if (hasEdgeConflict(solutions[i], solutions[j]))
                    return true;
            }
        }
        return false;
    }

    static Conflict getFirstConflict(const TreeNode &P) {
        auto solutions = P.solution;

        for (size_t i = 0; i < solutions.size(); i++) {
            for (size_t j = i + 1; j < solutions.size(); j++) {
                size_t max_index = std::max(solutions[i].size(), solutions[j].size());
                for (size_t k = 0; k < max_index; k++) {
                    Cell pos1 = (k < solutions[i].size()) ? solutions[i][k] : solutions[i].back();
                    Cell pos2 = (k < solutions[j].size()) ? solutions[j][k] : solutions[j].back();

                    if (pos1 == pos2) {
                        return Conflict(i, j, k, pos1, pos1); 
                    }
                }
            }
        }

        for (size_t i = 0; i < solutions.size(); i++) {
            for (size_t j = i + 1; j < solutions.size(); j++) {
                size_t min_index = std::min(solutions[i].size(), solutions[j].size());
                if (min_index < 2) continue;

                for (size_t k = 0; k < min_index - 1; k++) {
                    if (solutions[i][k] == solutions[j][k + 1] && solutions[i][k + 1] == solutions[j][k]) {
                        return Conflict(i, j, k + 1, solutions[i][k + 1], solutions[j][k + 1]);
                    }
                }
            }
        }

        return Conflict(0, 0, 0, Cell(0, 0), Cell(0, 0));
    }

};