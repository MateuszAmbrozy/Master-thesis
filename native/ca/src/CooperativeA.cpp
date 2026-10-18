#include "CooperativeA.h"
#include <climits>
#include <chrono>
#include <algorithm>

CooperativeA::CooperativeA(map::Graph graph)
: A::Astar(graph)
{}

std::vector<CA::Cell> CooperativeA::getNeighbors(const Agent& agent, map::Cell target,  CA::Node &current, const Reservation &table, std::vector<std::vector<bool>>& closedList) {
    std::vector<CA::Cell> neighbors;
    std::vector<map::Cell> illicits = agent.getIllicits();
    const std::vector<std::pair<int, int>> moves{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    for (auto dir : moves)
    {
        CA::Cell neighbor(current.x + dir.first, current.y + dir.second, current.t + 1);
        bool containtsIllicits = std::find(illicits.begin(), illicits.end(), neighbor) != illicits.end();

        if (isValid(neighbor.x, neighbor.y) && !containtsIllicits)
        {
            if (!table.isReserved(neighbor.x, neighbor.y, neighbor.t) &&
                !table.wouldCauseEdgeCollision(current.x, current.y, current.t, neighbor.x, neighbor.y))
            {
                neighbors.push_back(neighbor);
            }
        }
    }

    CA::Cell waitNeighbor(current.x, current.y, current.t + 1);
    if (!table.isReserved(waitNeighbor.x, waitNeighbor.y, waitNeighbor.t)) 
    {
        neighbors.push_back(waitNeighbor);
    }

    return neighbors;
}

std::vector<CA::Cell> CooperativeA::pathToTarget(const Agent& unit, const map::Cell& start, int currentTime, const map::Cell& target, Reservation& table)
{
    std::vector<CA::Cell> path;

    if (!isValid(target.x, target.y)) {
        std::cerr << "Destination is an obstacle or is outside the map." << std::endl;
        return path;
    }

    if (isDestination(start.x, start.y, target.x, target.y)) {
        return {CA::Cell(target.x, target.y, currentTime)};
    }
    
    std::vector<map::Cell> illicits = unit.getIllicits();
    if(std::find(illicits.begin(), illicits.end(), target) != illicits.end()) {
        std::cerr << "Destination is a forbidden cell (illicit)." << std::endl;
        return path;
    }

    std::vector<std::vector<bool>> closedList(graph.cells.size(), std::vector<bool>(graph.cells[0].size(), false));
    std::unordered_map<CA::Node, unsigned int> closedSet;
    std::priority_queue<CA::Node, std::vector<CA::Node>, A::CompareCells> openList;

    CA::Node startNode(currentTime, start.x, start.y, start.isObstacle);
    startNode.gCost = 0;
    startNode.hCost = CA::Node::calculateH(start.x, start.y, target.x, target.y);
    startNode.fCost = startNode.gCost + startNode.hCost;
    startNode.parent = nullptr;

    openList.push(startNode);
    CA::Node currentCell;

    while (!openList.empty())
    {
        currentCell = openList.top();
        openList.pop();
        closedSet.insert({currentCell, currentCell.fCost});

        if (isDestination(currentCell.x, currentCell.y, target.x, target.y)) 
        {
            std::shared_ptr<CA::Node> temp = std::make_shared<CA::Node>(currentCell);
            while (temp != nullptr) 
            {
                path.push_back(CA::Cell(temp->x, temp->y, temp->t));
                temp = temp->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
    
        std::vector<CA::Cell> neighbors = getNeighbors(unit, target, currentCell, table, closedList);

        for (CA::Cell& neighbor : neighbors)
        {
            double gNew = currentCell.gCost + 1;
            double hNew = CA::Node::calculateH(neighbor.x, neighbor.y, target.x, target.y);
            double fNew = gNew + hNew;

            auto parentNode = std::make_shared<CA::Node>(currentCell);
            CA::Node neighborCell(neighbor.t, neighbor.x, neighbor.y, false, parentNode, gNew, hNew, fNew);
            
            auto it = closedSet.find(neighborCell);
            if (it != closedSet.end()) {
                if (fNew >= it->second) continue;
            } else {
                closedSet[neighborCell] = fNew;
            }

            openList.push(neighborCell);
        }
    }
    
    std::cerr << "No path to the destination found for agent " << unit.getId() << std::endl;
    return {};
}


std::vector<CA::Cell> CooperativeA::findPath(Agent& unit, int currentTime, Reservation& table) 
{    
    map::Cell startPos = unit.getStartPosition(); 
    map::Cell goalPos = unit.getGoal();

    std::vector<CA::Cell> fullPath = pathToTarget(unit, startPos, currentTime, goalPos, table);

    if(fullPath.empty()) {
        return {};
    }

    
    for(size_t i = 0; i < fullPath.size(); i++)
    {            
        table.reserve(fullPath[i].x, fullPath[i].y, fullPath[i].t);
        
        
        if(i < fullPath.size() - 1) {
            table.reserveEdge(fullPath[i].x, fullPath[i].y, fullPath[i + 1].x, fullPath[i + 1].y, fullPath[i].t);
        }
    }

    
    int goalArrivalTime = fullPath.back().t;
    int MAX_RESERVATION_TIME = 1000;
    
    for (int t = goalArrivalTime + 1; t < goalArrivalTime + MAX_RESERVATION_TIME; ++t) {
        table.reserve(goalPos.x, goalPos.y, t);
    }

    return fullPath;
}