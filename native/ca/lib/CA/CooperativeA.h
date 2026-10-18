#pragma once
#include <vector>
#include <unordered_map>

#include "Cell.h"
#include "AStar.h"
#include "Graph.h"
#include "Agent.h"
#include "Reservation.h"

class CooperativeA : public A::Astar
{
private:
    
    std::vector<CA::Cell> getNeighbors(const Agent& agent, map::Cell target,  CA::Node &current, const Reservation &table, std::vector<std::vector<bool>>& closedList);
    
    std::vector<CA::Cell> pathToTarget(const Agent& unit, const map::Cell& start, int currentTime, const map::Cell& target, Reservation& table);

public:
    CooperativeA(map::Graph graph);
    
    std::vector<CA::Cell> findPath(Agent& unit, int currentTime, Reservation& table);
};