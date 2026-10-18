#pragma once
#include <iostream>
#include <queue>
#include "ConstraintTree.h"
#include "util.h"

class HighLevel
{
private:
    Map map;
    std::unique_ptr<ConstraintTree> CT;
    std::priority_queue<std::shared_ptr<TreeNode>, std::vector<std::shared_ptr<TreeNode>>, CompareNodes> openList;
    std::shared_ptr<TreeNode> root = std::make_shared<TreeNode>();
public:
    HighLevel(Map& map, const std::vector<Constraint>& constraints);
    std::vector<std::vector<Cell>> solve(double timeout = 10.0);
};