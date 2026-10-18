#include "Agent.h"

Agent::Agent(int id, map::Cell startPos, map::Cell goalPos)
    : id(id), startPos(startPos) , goalPos(goalPos), currentPos(startPos)
         {}

int Agent::getId() const {
    return id;
}

map::Cell Agent::getStartPosition() const {
    return startPos;
}

map::Cell Agent::getGoal() const {
    return goalPos;
}

std::vector<map::Cell> Agent::getIllicits() const
{
    return illicits;
}

void Agent::addIllicitCell(const map::Cell& cell)
{
    illicits.push_back(cell);
}

bool Agent::operator ==(const Agent& other) const
{
    return id==other.id &&
           startPos == other.startPos &&
           goalPos == other.goalPos &&
           illicits == other.illicits;
}
