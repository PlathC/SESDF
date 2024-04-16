#include "bcs/core/loader.hpp"

#include "bcs/core/molecule.hpp"

//
#include <fmt/printf.h>
#pragma warning( push, 0 )
#include <chemfiles.hpp>
#pragma warning( pop )
#include <magic_enum/magic_enum.hpp>

namespace bcs
{
    Molecule load( const Path & path )
    {
        std::string extension = path.extension().string().substr( 1, path.extension().string().size() );
        std::transform( extension.begin(), extension.end(), extension.begin(), toupper );

        static bool configured = false;
        if ( !configured )
        {
            configured = true;
#ifndef NDEBUG
            chemfiles::warning_callback_t callback = []( const std::string & p_log ) { fmt::print( "{}\n", p_log ); };
#else
            chemfiles::warning_callback_t callback = []( const std::string & p_log ) { /*fmt::print( p_log );*/ };
#endif
            chemfiles::set_warning_callback( callback );
        }

        chemfiles::Trajectory trajectory { path.string() };
        if ( trajectory.nsteps() == 0 )
            throw std::runtime_error( "Trajectory is empty" );

        chemfiles::Frame                        frame    = trajectory.read();
        const chemfiles::Topology &             topology = frame.topology();
        const std::vector<chemfiles::Residue> & residues = topology.residues();
        const std::vector<chemfiles::Bond> &    bonds    = topology.bonds();

        if ( frame.size() != topology.size() )
            throw std::runtime_error( "Data count missmatch" );

        // Set molecule properties.
        std::string name;
        if ( frame.get( "name" ) )
            name = frame.get( "name" )->as_string();

        Molecule molecule {};
        for ( const chemfiles::Residue & residue : residues )
        {
            for ( const std::size_t atomId : residue )
            {
                const chemfiles::Atom & atom = topology[ atomId ];

                const chemfiles::span<chemfiles::Vector3D> & positions = frame.positions();
                const chemfiles::Vector3D &                  position  = positions[ atomId ];

                std::string atomSymbol = atom.type();
                std::transform( atomSymbol.begin(),
                                atomSymbol.end(),
                                atomSymbol.begin(),
                                []( unsigned char c ) { return std::toupper( c ); } );
                std::optional symbol = magic_enum::enum_cast<SYMBOL>( "A_" + atomSymbol );
                symbol               = symbol ? symbol : SYMBOL::UNKNOWN;

                molecule.emplace_back( Vec4f { position[ 0 ], position[ 1 ], position[ 2 ], getRadius( *symbol ) } );
            }
        }

        return molecule;
    }
} // namespace bcs
