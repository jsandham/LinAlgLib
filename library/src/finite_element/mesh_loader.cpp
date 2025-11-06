#include "../../include/finite_element/mesh_loader.h"
#include "../../include/finite_element/mesh.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace linalg
{
    static bool parseMeshFormat(std::stringstream& ss, std::pair<double, int>& mesh_format)
    {
        double version;
        int    file_type;
        int    data_size;

        if(!(ss >> version >> file_type >> data_size))
        {
            std::cerr << "Error reading $MeshFormat header line." << std::endl;
            return false;
        }

        if(version != 2.2)
        {
            std::cerr << "Warning: File version is " << version << ". Expected 2.2." << std::endl;
            return false;
        }

        // We only support ASCII (file_type = 0) in this implementation.
        if(file_type == 1)
        {
            std::cerr << "Error: Binary file format (type 1) is not supported by this loader."
                      << std::endl;
            return false;
        }

        std::string end_tag;
        ss >> end_tag; // Consume $EndMeshFormat
        if(end_tag != "$EndMeshFormat")
        {
            std::cerr << "Error: Missing $EndMeshFormat tag." << std::endl;
            return false;
        }

        mesh_format = {version, file_type};
        return true;
    }

    static bool parseNodes(std::stringstream& ss, mesh& mesh)
    {
        int num_nodes;
        if(!(ss >> num_nodes))
        {
            std::cerr << "Error: Unable to read number of nodes." << std::endl;
            return false;
        }

        mesh.nodes.reserve(num_nodes);
        for(int i = 0; i < num_nodes; ++i)
        {
            node node;
            if(!(ss >> node.tag >> node.x >> node.y >> node.z))
            {
                std::cout << "Error: Unable to read node data." << std::endl;
                return false;
            }
            mesh.nodes.push_back(node);
        }

        std::string end_tag;
        ss >> end_tag; // Consume $EndNodes
        if(end_tag != "$EndNodes")
        {
            std::cerr << "Error: Missing $EndNodes tag." << std::endl;
            return false;
        }

        return true;
    }

    static bool parseElements(std::stringstream& ss, mesh& mesh)
    {
        int num_elements;
        if(!(ss >> num_elements))
        {
            std::cerr << "Error: Unable to read number of elements." << std::endl;
            return false;
        }

        mesh.elements.reserve(num_elements);
        for(int i = 0; i < num_elements; ++i)
        {
            element element;
            int     num_tags;

            // Read basic element info: tag, type, and number of tags
            if(!(ss >> element.tag >> element.type >> num_tags))
            {
                std::cerr << "Error: Unable to read element header data." << std::endl;
                return false;
            }

            // Read tags (Physical tag, Elementary tag, etc.)
            element.tags.resize(num_tags);
            for(int j = 0; j < num_tags; ++j)
            {
                if(!(ss >> element.tags[j]))
                {
                    std::cerr << "Error: Unable to read element tags." << std::endl;
                    return false;
                }
            }

            // Determine expected number of nodes for the element type
            // This is a simplified lookup; a real implementation would use a comprehensive table.
            int num_nodes;
            switch(element.type)
            {
            case 1:
                num_nodes = 2;
                break; // 2-node line
            case 2:
                num_nodes = 3;
                break; // 3-node triangle
            case 3:
                num_nodes = 4;
                break; // 4-node quad
            case 4:
                num_nodes = 4;
                break; // 4-node tetra
            case 5:
                num_nodes = 8;
                break; // 8-node hexa
            // ... add other types as needed
            default:
                // Note: Gmsh has many types; for safety, we assume a max number of nodes for unhandled types
                // For MSH 2.2, connectivity is always after tags.
                // We must read until the end of the line or until we reach the max connectivity size.
                // Since we can't reliably know the node count without a full table, we'll use a safer approach:
                // Read the known number of nodes for common types and throw for unknown for this example.
                std::cerr << "Warning: Unsupported Gmsh element type " << element.type
                          << ". Attempting to read connectivity." << std::endl;
                return false;
            }

            // Read node connectivity
            element.node_tags.resize(num_nodes);
            for(int j = 0; j < num_nodes; ++j)
            {
                if(!(ss >> element.node_tags[j]))
                {
                    std::cerr << "Error: Unable to read element node connectivity." << std::endl;
                    return false;
                }
            }
            mesh.elements.push_back(element);
        }

        std::string end_tag;
        ss >> end_tag; // Consume $EndElements
        if(end_tag != "$EndElements")
        {
            std::cerr << "Error: Missing $EndElements tag." << std::endl;
            return false;
        }

        return true;
    }
}

bool linalg::load_gmsh_mesh(const std::string& filename, mesh& mesh)
{
    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "Error: Unable to open mesh file " << filename << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::stringstream ss(buffer.str());
    std::string       line;

    while(ss >> line)
    {
        if(line == "$MeshFormat")
        {
            std::pair<double, int> mesh_format;
            if(!parseMeshFormat(ss, mesh_format))
            {
                return false;
            }
        }
        else if(line == "$Nodes")
        {
            if(!parseNodes(ss, mesh))
            {
                return false;
            }
        }
        else if(line == "$Elements")
        {
            if(!parseElements(ss, mesh))
            {
                return false;
            }
        }
        else if(line.rfind("$", 0) == 0)
        { // Check if it's another $Section tag
            // Skip unknown or unneeded sections like $ElementData, $PhysicalNames
            std::string end_tag = "$End" + line.substr(1);
            std::string current_token;
            while(ss >> current_token && current_token != end_tag)
            {
                // Keep skipping tokens until the end tag is found
            }
        }
    }
    return true;
}