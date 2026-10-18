#include "lib/LowLevel.h"
#include "lib/HighLevel.h"
#include "lib/util.h"
#include <iostream>

void lowLevelTest(LowLevel lowLevel, std::vector<Constraint> constraints, Map map)
{ 
    std::vector<std::vector<Cell>> result = lowLevel.getPaths(constraints, map);
    for(int i=0; i<result.size(); i++)
    {
        for(int j=0; j<result[i].size(); j++)
        {
            std::cout<<"(" << result[i][j].position.x << ", " <<  result[i][j].position.y << ")" << std::endl;
        }
    }
}
Map createMap(int mapWidth, int mapHeight)
{
    Map map;
    map.cells.resize(mapWidth, std::vector<Cell>(mapHeight, Cell(Position(0, 0), 0, 0)));
    
    for (int i = 0; i < mapWidth; ++i) {
        for (int j = 0; j < mapHeight; ++j) {
            map.cells[i][j] = Cell(Position(i, j));
        }
    }

    map.cells[0][0].isObstacle = true; // D1 (0, 0)
    map.cells[0][3].isObstacle = true; // D3 (0, 2)

    map.cells[3][0].isObstacle = true; // C2 (2, 1)
    map.cells[3][3].isObstacle = true; // B2 (1, 1)


    
    Position startAgent1(1, 0); // D1
    Position goalAgent1(2, 3);  // B3
    Agent agent1(1, startAgent1, goalAgent1);
    map.agents.push_back(agent1);

    Position startAgent2(0, 1); // C1
    Position goalAgent2(3, 2);  // A4
    Agent agent2(2, startAgent2, goalAgent2);
    map.agents.push_back(agent2);

    return map;
}

int main() {
    int mapWidth = 4;
    int mapHeight = 4;
    
    Map map = createMap(mapWidth, mapHeight);
    map.cells.resize(mapWidth, std::vector<Cell>(mapHeight, Cell(Position(0, 0), 0, 0)));

    std::vector<Constraint> constraints;

    LowLevel lowLevel;
    HighLevel highLevel(map, constraints);

    std::cout<<"Printing paths...\n";
    std::vector<std::vector<Cell>> result = highLevel.solve();

    for(int i=0; i<result.size(); i++)
    {
        std::cout<<"Agent " << i+1 << std::endl;
        for(int j=0; j<result[i].size(); j++)
        {
            std::cout<<"(" << result[i][j].position.x << ", " <<  result[i][j].position.y << ")"  << ", t: " << j << std::endl;
        }
    }

    return 0;
}