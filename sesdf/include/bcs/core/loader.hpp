#ifndef BCS_CORE_LOADER_HPP
#define BCS_CORE_LOADER_HPP

#include <filesystem>

#include "bcs/core/molecule.hpp"

namespace bcs
{
    using Path = std::filesystem::path;
    Molecule load( const Path & path );
} // namespace bcs

#endif // BCS_LOADER_HPP
