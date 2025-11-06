//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************

#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>

/*! \file
 *  \brief mesh.h provides classes for mesh representation
 */

namespace linalg
{
    struct node
    {
        int    tag;
        double x, y, z;
    };

    /**
 * @brief Represents a single element (e.g., triangle, tetrahedron) in the mesh.
 */
    struct element
    {
        int              tag;
        int              type; // Gmsh element type (e.g., 2 for 3-node triangle)
        std::vector<int> tags; // Physical and elementary tags
        std::vector<int> node_tags; // Connectivity: list of connected node tags
    };

    /**
 * @brief Container for the entire mesh data.
 */
    struct mesh
    {
        std::vector<node>    nodes;
        std::vector<element> elements;
    };
} // namespace linalg

#endif // MESH_H