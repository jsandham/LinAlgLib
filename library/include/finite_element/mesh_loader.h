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

#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <string>
#include <vector>

#include "mesh.h"

/*! \file
 *  \brief mesh_loader.h provides functions for loading mesh data from files
 */

namespace linalg
{
    /**
 * @brief Loads a mesh from a Gmsh .msh file.
 *
 * @param filename The path to the .msh file.
 * @param mesh The Mesh object to populate with the loaded data.
 * @return true if the mesh was loaded successfully, false otherwise.
 */
    bool load_gmsh_mesh(const std::string& filename, mesh& mesh);
} // namespace linalg

#endif // MESH_LOADER_H