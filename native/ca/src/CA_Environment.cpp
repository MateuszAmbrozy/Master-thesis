#include "CA_Environment.h"

CA_Environment::CA_Environment(std::vector<Agent> agents, map::Graph graph)
: agents(agents), sta(graph)
{
}


void CA_Environment::MOVEAGENTS(int timestep)
{
    for(auto& agent : agents)
    {
        std::vector<CA::Cell> path = agent.getPath();
        if (path.empty()) continue;

        
        bool found = false;
        for (const auto& cell : path)
        {
            if (cell.t == timestep)
            {
                agent.move(map::Cell(cell.x, cell.y));
                found = true;
                break;
            }
        }
        
        
        if (!found && timestep > path.back().t) {
            agent.move(map::Cell(path.back().x, path.back().y));
        }
    }
}
void CA_Environment::planAllPaths()
{
    
    for (auto& agent : agents)
    {
        
        std::vector<CA::Cell> path = sta.findPath(agent, 0, table);
        
        if (!path.empty()) {
            agent.assignPath(path);
        } else {
            std::cerr << "Could not find a path for agent " << agent.getId() << "\n";
        }
    }
}

void CA_Environment::runTimestep(int timestep)
{
    MOVEAGENTS(timestep);
}
