/**
 * @file Environment.h
 * @author 
 * @brief Cooperative A* Environment 
 * @version 0.1
 * @date 2025-06-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once
#include <iostream>
#include <vector>
#include <optional>
#include <ctime>
#include <unordered_map>
#include "Agent.h"

#include "Graph.h"

#include "Reservation.h"
#include "CooperativeA.h"



class CA_Environment
{
private:
    std::vector<Agent> agents;
    CooperativeA sta;
    Reservation table;

public:
    CA_Environment(std::vector<Agent> agents, map::Graph graph);
    


    std::vector<Agent> getAgents() {return this->agents;}

    bool allTasksCompleted()
    {
        for (const auto& agent : agents)
        {
            if (!(agent.getPosition() == agent.getGoal()))
            {
                return false;
            }
        }
        return true;
    }
    void planAllPaths();
    void solve()
    {
        std::cout << "--- Starting path planning ---\n";
        planAllPaths();

        std::cout << "--- Starting simulation ---\n";
        int timestep = 0;
        const int MAX_TIME = 500; 

        while (!allTasksCompleted() && timestep < MAX_TIME)
        {
            MOVEAGENTS(timestep);
            
            std::cout << "T=" << timestep << " | ";
            for(const auto& a : agents) {
                std::cout << "A" << a.getId() << ":(" << a.getPosition().x << "," << a.getPosition().y << ") ";
            }
            std::cout << "\n";

            timestep++;
        }

        if (allTasksCompleted()) {
            std::cout << "--- Success: All agents reached their goals at timestep " << timestep - 1 << " ---\n";
        } else {
            std::cout << "--- Timeout: The mission did not finish within the time limit ---\n";
        }
    }
    void MOVEAGENTS(int timestep);
    void addAgent(Agent& newAgent) {agents.push_back(newAgent);}

    void runTimestep(int timestep);
};
