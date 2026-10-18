/**
 * @file Agent.h
 * @author 
 * @brief Agent background class storing all necessery variables
 * @version 0.1
 * @date 2025-06-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once
#include "Cell.h"
#include "Graph.h"
#include <vector>


class Agent {
private:
    int id;
    map::Cell currentPos;
    map::Cell startPos;
    map::Cell goalPos;
    std::vector<CA::Cell> path;
    std::vector<map::Cell> illicits;

public:
    Agent(int id, map::Cell startPos, map::Cell goalPos);
    Agent(const Agent& other) = default;
    ~Agent() = default;

    int getId() const;
    
    map::Cell getPosition() const { return this->currentPos; }
    void move(map::Cell new_position) { this->currentPos = new_position; }
    map::Cell getStartPosition() const;
    map::Cell getGoal() const;
    std::vector<map::Cell> getIllicits() const;

    void assignPath(const std::vector<CA::Cell>& p) {this->path = p;}
    std::vector<CA::Cell> getPath() const { return this->path;}

    void addIllicitCell(const map::Cell& cell);
    bool operator == (const Agent& other) const; 
};
