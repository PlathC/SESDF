#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <GL/gl3w.h>

//
#include <ContourBuildup/BufferObject.hpp>
#include <ContourBuildup/Texture2D.hpp>
#include <ContourBuildup/particleSystem.cuh>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <vector_functions.h>

#include "bcs/contour_buildup/contour_buildup.hpp"
#include "bcs/core/molecule.hpp"

namespace bcs
{
    // This function returns the best GPU (with maximum GFLOPS)
    int cudaUtilGetMaxGflopsDeviceId()
    {
        int device_count = 0;
        cudaGetDeviceCount( &device_count );

        cudaDeviceProp device_properties;
        int            max_gflops_device = 0;
        int            max_gflops        = 0;

        int current_device = 0;
        cudaGetDeviceProperties( &device_properties, current_device );
        max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
        ++current_device;

        while ( current_device < device_count )
        {
            cudaGetDeviceProperties( &device_properties, current_device );
            int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
            if ( gflops > max_gflops )
            {
                max_gflops        = gflops;
                max_gflops_device = current_device;
            }
            ++current_device;
        }

        return max_gflops_device;
    }

    ContourBuildup::ContourBuildup( const ConstSpan<Vec4f> molecule,
                                    const Aabb &           aabb,
                                    const float            probeRadius,
                                    uint32_t               gridDimensions,
                                    bool                   buildSurface ) :
        _molecule( molecule ),
        atomNeighborCount( 64 ), probeRadius( probeRadius ), probeNeighborCount( 32 )
    {
        this->numAtoms = _molecule.size;

        if ( setCUDAGLDevice )
        {
            cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId() );
#ifndef NDEBUG
            printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError() ) );
#endif // NDEBUG
            setCUDAGLDevice = false;
        }

        // set grid dimensions
        this->gridSize     = glm::uvec3( gridDimensions );
        this->numGridCells = this->gridSize.x * this->gridSize.y * this->gridSize.z;

        glm::vec3 size      = aabb.max - aabb.min;
        glm::vec3 worldSize = glm::vec3( size.x, size.y, size.z );
        this->gridSortBits  = 18; // increase this for larger grids

        // set parameters
        this->params.gridSize  = make_uint3( this->gridSize.x, this->gridSize.y, this->gridSize.z );
        this->params.numCells  = this->numGridCells;
        this->params.numBodies = this->numAtoms;
        // this->params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);

        this->params.worldOrigin = make_float3( aabb.min.x, aabb.min.y, aabb.min.z );
        this->params.cellSize    = make_float3(
            worldSize.x / this->gridSize.x, worldSize.y / this->gridSize.y, worldSize.z / this->gridSize.z );
        this->params.probeRadius     = this->probeRadius;
        this->params.maxNumNeighbors = this->atomNeighborCount;

        // allocate host storage
        // hPos_.clear();
        // hPos_.resize( this->numAtoms, glm::vec4( 0.0f ) );

        // allocate GPU data
        unsigned int memSize = sizeof( float ) * 4 * this->numAtoms;
        // array for atom positions
        allocateArray( (void **)&m_dPos, memSize );
        // cudaMalloc(  (void**)&m_dPos, memSize);
        // cudaError e;
        // e = cudaGetLastError();

        // cuMemGetInfo
        // uint free, total;
        // cuMemGetInfo( &free, &total);
        // megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
        //     "Free GPU Memory: %i / %i (MB)", free / ( 1024 * 1024), total / ( 1024 * 1024));
        //  array for sorted atom positions
        allocateArray( (void **)&m_dSortedPos, memSize );
        // array for sorted atom positions
        allocateArray( (void **)&m_dSortedProbePos, memSize * this->atomNeighborCount );
        // array for the counted number of atoms
        allocateArray( (void **)&m_dNeighborCount, this->numAtoms * sizeof( uint ) );
        // array for the neighbor atoms
        allocateArray( (void **)&m_dNeighbors, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );
        // array for the small circles
        allocateArray( (void **)&m_dSmallCircles, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ) );
        // array for the small circle visibility
        allocateArray( (void **)&m_dSmallCircleVisible, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );
        // array for the small circle visibility prefix sum
        allocateArray( (void **)&m_dSmallCircleVisibleScan, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );
        
        // Avoid overflow during calculation
        const std::size_t arcCount = this->numAtoms * this->atomNeighborCount * this->atomNeighborCount;
        // array for the arcs
        allocateArray( (void **)&m_dArcs, arcCount * 4 * sizeof( float ) );
        // array for the arcs
        allocateArray( (void **)&m_dArcIdxK, arcCount * sizeof( uint ) );
        // array for the arc count
        allocateArray( (void **)&m_dArcCount, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );
        // array for the arc count scan (prefix sum)
        allocateArray( (void **)&m_dArcCountScan, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );

        allocateArray( (void **)&m_dGridParticleHash, this->numAtoms * sizeof( uint ) );
        allocateArray( (void **)&m_dGridParticleIndex, this->numAtoms * sizeof( uint ) );

        allocateArray( (void **)&m_dGridProbeHash, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );
        allocateArray( (void **)&m_dGridProbeIndex, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );

        allocateArray( (void **)&m_dCellStart, this->numGridCells * sizeof( uint ) );
        allocateArray( (void **)&m_dCellEnd, this->numGridCells * sizeof( uint ) );

        // clear all buffers
        for ( auto & e : buffers_ )
        {
            e = nullptr;
        }

        // re-create all buffers
        buffers_[ static_cast<int>( Buffers::PROBE_POS ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->getName() );
        getLastCudaError( "init failed" );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->getName() );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->getName() );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->getName() );

        buffers_[ static_cast<int>( Buffers::TORUS_POS ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->getName() );

        buffers_[ static_cast<int>( Buffers::TORUS_VS ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->getName() );

        buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->getName() );

        // get maximum texture size
        GLint texSize;
        glGetIntegerv( GL_MAX_TEXTURE_SIZE, &texSize );
        texHeight            = std::min( this->numAtoms * 3, static_cast<uint>( texSize ) );
        texWidth             = this->probeNeighborCount * ( ( this->numAtoms * 3 ) / texSize + 1 );
        this->params.texSize = texSize;

        // create singularity texture
        std::vector<std::pair<GLenum, GLint>>   int_parameters = { { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE },
                                                                   { GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE },
                                                                   { GL_TEXTURE_MIN_FILTER, GL_NEAREST },
                                                                   { GL_TEXTURE_MAG_FILTER, GL_NEAREST } };
        std::vector<std::pair<GLenum, GLfloat>> float_parameters;
        glowl::TextureLayout                    tx_layout { GL_RGB32F,
                                         static_cast<int>( texWidth ),
                                         static_cast<int>( texHeight ),
                                         1,
                                         GL_RGB,
                                         GL_FLOAT,
                                         1,
                                         int_parameters,
                                         float_parameters };
        singTex_ = std::make_unique<glowl::Texture2D>( "molecule_cbc_singTex", tx_layout, nullptr );

        // create PBO
        buffers_[ static_cast<int>( Buffers::SING_TEX ) ] = std::make_unique<glowl::BufferObject>(
            GL_PIXEL_UNPACK_BUFFER, nullptr, this->texWidth * texHeight * 3 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->getName() );

        // create texture coordinate buffer object
        buffers_[ static_cast<int>( Buffers::TEX_COORD ) ] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, nullptr, this->numAtoms * 3 * 3 * sizeof( float ), GL_DYNAMIC_DRAW );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->getName() );

        // set parameters
        setParameters( &this->params );

        // create VAOs
        if ( sphericalTriangleVAO_ == 0 )
        {
            glGenVertexArrays( 1, &sphericalTriangleVAO_ );
        }
        glBindVertexArray( sphericalTriangleVAO_ );

        buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->bind();
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->bind();
        glEnableVertexAttribArray( 1 );
        glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->bind();
        glEnableVertexAttribArray( 2 );
        glVertexAttribPointer( 2, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->bind();
        glEnableVertexAttribArray( 3 );
        glVertexAttribPointer( 3, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->bind();
        glEnableVertexAttribArray( 4 );
        glVertexAttribPointer( 4, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

        glBindVertexArray( 0 );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glDisableVertexAttribArray( 0 );
        glDisableVertexAttribArray( 1 );
        glDisableVertexAttribArray( 2 );
        glDisableVertexAttribArray( 3 );
        glDisableVertexAttribArray( 4 );

        if ( torusVAO_ == 0 )
        {
            glGenVertexArrays( 1, &torusVAO_ );
        }
        glBindVertexArray( torusVAO_ );

        buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->bind();
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->bind();
        glEnableVertexAttribArray( 1 );
        glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->bind();
        glEnableVertexAttribArray( 2 );
        glVertexAttribPointer( 2, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

        glBindVertexArray( 0 );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glDisableVertexAttribArray( 0 );
        glDisableVertexAttribArray( 1 );
        glDisableVertexAttribArray( 2 );

        cudaInitalized = true;

        if ( buildSurface )
            build();
    }

    ContourBuildup::ContourBuildup( ContourBuildup && other ) noexcept
    {
        std::swap( _molecule, other._molecule );
        std::swap( sphereVAO_, other.sphereVAO_ );
        std::swap( torusVAO_, other.torusVAO_ );
        std::swap( sphericalTriangleVAO_, other.sphericalTriangleVAO_ );
        std::swap( probeRadius, other.probeRadius );
        std::swap( atomNeighborCount, other.atomNeighborCount );
        std::swap( cudaInitalized, other.cudaInitalized );
        std::swap( numAtoms, other.numAtoms );
        std::swap( params, other.params );
        std::swap( gridSize, other.gridSize );
        std::swap( numGridCells, other.numGridCells );
        std::swap( m_dPos, other.m_dPos );
        std::swap( m_dSortedPos, other.m_dSortedPos );
        std::swap( m_dSortedProbePos, other.m_dSortedProbePos );
        std::swap( m_dNeighborCount, other.m_dNeighborCount );
        std::swap( m_dNeighbors, other.m_dNeighbors );
        std::swap( m_dSmallCircles, other.m_dSmallCircles );
        std::swap( m_dSmallCircleVisible, other.m_dSmallCircleVisible );
        std::swap( m_dSmallCircleVisibleScan, other.m_dSmallCircleVisibleScan );
        std::swap( m_dArcs, other.m_dArcs );
        std::swap( m_dArcIdxK, other.m_dArcIdxK );
        std::swap( m_dArcCount, other.m_dArcCount );
        std::swap( m_dArcCountScan, other.m_dArcCountScan );
        std::swap( m_dGridParticleHash, other.m_dGridParticleHash );
        std::swap( m_dGridParticleIndex, other.m_dGridParticleIndex );
        std::swap( m_dGridProbeHash, other.m_dGridProbeHash );
        std::swap( m_dGridProbeIndex, other.m_dGridProbeIndex );
        std::swap( m_dCellStart, other.m_dCellStart );
        std::swap( m_dCellEnd, other.m_dCellEnd );
        std::swap( gridSortBits, other.gridSortBits );
        std::swap( numProbes, other.numProbes );
        std::swap( numSC, other.numSC );
        std::swap( buffers_, other.buffers_ );
        std::swap( singTex_, other.singTex_ );
        std::swap( probeNeighborCount, other.probeNeighborCount );
        std::swap( texHeight, other.texHeight );
        std::swap( texWidth, other.texWidth );
        std::swap( width, other.width );
        std::swap( height, other.height );
        std::swap( setCUDAGLDevice, other.setCUDAGLDevice );
    }

    ContourBuildup & ContourBuildup::operator=( ContourBuildup && other ) noexcept
    {
        std::swap( _molecule, other._molecule );
        std::swap( sphereVAO_, other.sphereVAO_ );
        std::swap( torusVAO_, other.torusVAO_ );
        std::swap( sphericalTriangleVAO_, other.sphericalTriangleVAO_ );
        std::swap( probeRadius, other.probeRadius );
        std::swap( atomNeighborCount, other.atomNeighborCount );
        std::swap( cudaInitalized, other.cudaInitalized );
        std::swap( numAtoms, other.numAtoms );
        std::swap( params, other.params );
        std::swap( gridSize, other.gridSize );
        std::swap( numGridCells, other.numGridCells );
        std::swap( m_dPos, other.m_dPos );
        std::swap( m_dSortedPos, other.m_dSortedPos );
        std::swap( m_dSortedProbePos, other.m_dSortedProbePos );
        std::swap( m_dNeighborCount, other.m_dNeighborCount );
        std::swap( m_dNeighbors, other.m_dNeighbors );
        std::swap( m_dSmallCircles, other.m_dSmallCircles );
        std::swap( m_dSmallCircleVisible, other.m_dSmallCircleVisible );
        std::swap( m_dSmallCircleVisibleScan, other.m_dSmallCircleVisibleScan );
        std::swap( m_dArcs, other.m_dArcs );
        std::swap( m_dArcIdxK, other.m_dArcIdxK );
        std::swap( m_dArcCount, other.m_dArcCount );
        std::swap( m_dArcCountScan, other.m_dArcCountScan );
        std::swap( m_dGridParticleHash, other.m_dGridParticleHash );
        std::swap( m_dGridParticleIndex, other.m_dGridParticleIndex );
        std::swap( m_dGridProbeHash, other.m_dGridProbeHash );
        std::swap( m_dGridProbeIndex, other.m_dGridProbeIndex );
        std::swap( m_dCellStart, other.m_dCellStart );
        std::swap( m_dCellEnd, other.m_dCellEnd );
        std::swap( gridSortBits, other.gridSortBits );
        std::swap( numProbes, other.numProbes );
        std::swap( numSC, other.numSC );
        std::swap( buffers_, other.buffers_ );
        std::swap( singTex_, other.singTex_ );
        std::swap( probeNeighborCount, other.probeNeighborCount );
        std::swap( texHeight, other.texHeight );
        std::swap( texWidth, other.texWidth );
        std::swap( width, other.width );
        std::swap( height, other.height );
        std::swap( setCUDAGLDevice, other.setCUDAGLDevice );

        return *this;
    }

    ContourBuildup::~ContourBuildup()
    {
        if ( this->cudaInitalized )
        {
            freeArray( m_dPos );
            freeArray( m_dSortedPos );
            freeArray( m_dSortedProbePos );
            freeArray( m_dNeighborCount );
            freeArray( m_dNeighbors );
            freeArray( m_dSmallCircles );
            freeArray( m_dSmallCircleVisible );
            freeArray( m_dSmallCircleVisibleScan );
            freeArray( m_dArcs );
            freeArray( m_dArcIdxK );
            freeArray( m_dArcCount );
            freeArray( m_dArcCountScan );
            freeArray( m_dGridParticleHash );
            freeArray( m_dGridParticleIndex );
            freeArray( m_dGridProbeHash );
            freeArray( m_dGridProbeIndex );
            freeArray( m_dCellStart );
            freeArray( m_dCellEnd );

            // Added to avoid "Already mapped" error
            for ( const auto & e : buffers_ )
            {
                cudaGLUnregisterBufferObject( e->getName() );
            }
        }
    }

    void ContourBuildup::build()
    {
        // write atom positions to VBO
        this->writeAtomPositionsVBO();

        // update constants
        this->params.probeRadius = this->probeRadius;
        setParameters( &this->params );

        // map OpenGL buffer object for writing from CUDA
        float * atomPosPtr;
        cudaGLMapBufferObject( (void **)&atomPosPtr, buffers_[ static_cast<int>( Buffers::ATOM_POS ) ]->getName() );

        // calculate grid hash
        calcHash( m_dGridParticleHash,
                  m_dGridParticleIndex,
                  // m_dPos,
                  atomPosPtr,
                  this->numAtoms );

        // sort particles based on hash
        sortParticles( m_dGridParticleHash, m_dGridParticleIndex, this->numAtoms );

        // reorder particle arrays into sorted order and
        // find start and end of each cell
        reorderDataAndFindCellStart( m_dCellStart,
                                     m_dCellEnd,
                                     m_dSortedPos,
                                     m_dGridParticleHash,
                                     m_dGridParticleIndex,
                                     // m_dPos,
                                     atomPosPtr,
                                     this->numAtoms,
                                     this->numGridCells );

        // unmap buffer object
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::ATOM_POS ) ]->getName() );

        // find neighbors of all atoms and compute small circles
        findNeighborsCB( m_dNeighborCount,
                         m_dNeighbors,
                         m_dSmallCircles,
                         m_dSortedPos,
                         m_dCellStart,
                         m_dCellEnd,
                         this->numAtoms,
                         this->atomNeighborCount,
                         this->numGridCells );

        // find and remove unnecessary small circles
        removeCoveredSmallCirclesCB( m_dSmallCircles,
                                     m_dSmallCircleVisible,
                                     m_dNeighborCount,
                                     m_dNeighbors,
                                     m_dSortedPos,
                                     numAtoms,
                                     params.maxNumNeighbors );

        cudaMemset( m_dArcCount, 0, this->numAtoms * this->atomNeighborCount * sizeof( uint ) );

        // compute all arcs for all small circles
        computeArcsCB( m_dSmallCircles,
                       m_dSmallCircleVisible,
                       m_dNeighborCount,
                       m_dNeighbors,
                       m_dSortedPos,
                       m_dArcs,
                       m_dArcCount,
                       numAtoms,
                       params.maxNumNeighbors );

        // ---------- vertex buffer object generation (for rendering) ----------
        // count total number of small circles
        scanParticles( m_dSmallCircleVisible, m_dSmallCircleVisibleScan, this->numAtoms * this->atomNeighborCount );
        // get total number of small circles
        numSC       = 0;
        uint lastSC = 0;
        checkCudaErrors(
            cudaMemcpy( (void *)&numSC,
                        (void *)( m_dSmallCircleVisibleScan + ( this->numAtoms * this->atomNeighborCount ) - 1 ),
                        sizeof( uint ),
                        cudaMemcpyDeviceToHost ) );
        checkCudaErrors(
            cudaMemcpy( (void *)&lastSC,
                        (void *)( m_dSmallCircleVisible + ( this->numAtoms * this->atomNeighborCount ) - 1 ),
                        sizeof( uint ),
                        cudaMemcpyDeviceToHost ) );
        numSC += lastSC;

        // count total number of arcs
        scanParticles( m_dArcCount, m_dArcCountScan, this->numAtoms * this->atomNeighborCount );
        // get total number of probes
        numProbes         = 0;
        uint lastProbeCnt = 0;
        checkCudaErrors( cudaMemcpy( (void *)&numProbes,
                                     (void *)( m_dArcCountScan + ( this->numAtoms * this->atomNeighborCount ) - 1 ),
                                     sizeof( uint ),
                                     cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy( (void *)&lastProbeCnt,
                                     (void *)( m_dArcCount + ( this->numAtoms * this->atomNeighborCount ) - 1 ),
                                     sizeof( uint ),
                                     cudaMemcpyDeviceToHost ) );
        numProbes += lastProbeCnt;

        // resize torus buffer objects
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->getName() );
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->getName() );
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->getName() );
        buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->rebuffer( nullptr, numSC * 4 * sizeof( float ) );
        buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->rebuffer( nullptr, numSC * 4 * sizeof( float ) );
        buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->rebuffer( nullptr, numSC * 4 * sizeof( float ) );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->getName() );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->getName() );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->getName() );

        // resize probe buffer object
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->getName() );
        buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->rebuffer( nullptr, numProbes * 4 * sizeof( float ) );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->getName() );

        // resize spherical triangle buffer objects
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->getName() );
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->getName() );
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->getName() );
        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->rebuffer( nullptr,
                                                                              numProbes * 4 * sizeof( float ) );
        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->rebuffer( nullptr,
                                                                              numProbes * 4 * sizeof( float ) );
        buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->rebuffer( nullptr,
                                                                              numProbes * 4 * sizeof( float ) );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->getName() );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->getName() );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->getName() );

        // map probe buffer object for writing from CUDA
        float * probePosPtr;
        cudaGLMapBufferObject( (void **)&probePosPtr, buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->getName() );

        // map spherical triangle buffer objects for writing from CUDA
        float *sphereTriaVec1Ptr, *sphereTriaVec2Ptr, *sphereTriaVec3Ptr;
        cudaGLMapBufferObject( (void **)&sphereTriaVec1Ptr,
                               buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->getName() );
        cudaGLMapBufferObject( (void **)&sphereTriaVec2Ptr,
                               buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->getName() );
        cudaGLMapBufferObject( (void **)&sphereTriaVec3Ptr,
                               buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->getName() );

        // map torus buffer objects for writing from CUDA
        float *torusPosPtr, *torusVSPtr, *torusAxisPtr;
        cudaGLMapBufferObject( (void **)&torusPosPtr, buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->getName() );
        cudaGLMapBufferObject( (void **)&torusVSPtr, buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->getName() );
        cudaGLMapBufferObject( (void **)&torusAxisPtr, buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->getName() );

        // compute vertex buffer objects for probe positions
        writeProbePositionsCB( probePosPtr,
                               sphereTriaVec1Ptr,
                               sphereTriaVec2Ptr,
                               sphereTriaVec3Ptr,
                               torusPosPtr,
                               torusVSPtr,
                               torusAxisPtr,
                               m_dNeighborCount,
                               m_dNeighbors,
                               m_dSortedPos,
                               m_dArcs,
                               m_dArcCount,
                               m_dArcCountScan,
                               m_dSmallCircleVisible,
                               m_dSmallCircleVisibleScan,
                               m_dSmallCircles,
                               this->numAtoms,
                               this->atomNeighborCount );

        // unmap torus buffer objects
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_POS ) ]->getName() );
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_VS ) ]->getName() );
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::TORUS_AXIS ) ]->getName() );

        // unmap spherical triangle buffer objects
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_1 ) ]->getName() );
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_2 ) ]->getName() );
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::SPHERE_TRIA_VEC_3 ) ]->getName() );

        // ---------- singularity handling ----------
        // calculate grid hash
        calcHash( m_dGridProbeHash, m_dGridProbeIndex, probePosPtr, numProbes );

        // sort probes based on hash
        sortParticles( m_dGridProbeHash, m_dGridProbeIndex, numProbes );

        // reorder particle arrays into sorted order and find start and end of each cell
        reorderDataAndFindCellStart( m_dCellStart,
                                     m_dCellEnd,
                                     m_dSortedProbePos,
                                     m_dGridProbeHash,
                                     m_dGridProbeIndex,
                                     probePosPtr,
                                     numProbes,
                                     this->numGridCells );

        // unmap probe buffer object
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::PROBE_POS ) ]->getName() );

        // resize texture coordinate buffer object
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->getName() );
        cudaGLUnregisterBufferObject( buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->getName() );
        buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->rebuffer( nullptr, numProbes * 3 * sizeof( float ) );
        buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->rebuffer(
            nullptr, numProbes * this->probeNeighborCount * 3 * sizeof( float ) );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->getName() );
        cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->getName() );

        // map texture coordinate buffer object for writing from CUDA
        float * texCoordPtr;
        cudaGLMapBufferObject( (void **)&texCoordPtr, buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->getName() );
        // map singularity texture buffer object for writing from CUDA
        float * singTexPtr;
        cudaGLMapBufferObject( (void **)&singTexPtr, buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->getName() );

        // find all intersecting probes for each probe and write them to a texture
        writeSingularityTextureCB( texCoordPtr,
                                   singTexPtr,
                                   m_dSortedProbePos,
                                   m_dGridProbeIndex,
                                   m_dCellStart,
                                   m_dCellEnd,
                                   numProbes,
                                   this->probeNeighborCount,
                                   this->numGridCells );

        // copyArrayFromDevice( m_hPos, m_dSortedProbePos, 0, sizeof(float)*4);
        // std::cout << "probe: " << m_hPos[0] << ", " << m_hPos[1] << ", " << m_hPos[2] << " r = " << m_hPos[3]
        // << std::endl; copyArrayFromDevice( m_hPos, singTexPtr, 0, sizeof(float)*3*this->probeNeighborCount);
        // for( unsigned int i = 0; i < this->probeNeighborCount; i++ ) {
        //     std::cout << "neighbor probe " << i << ": " << m_hPos[i*3] << " " << m_hPos[i*3+1] << " " <<
        //     m_hPos[i*3+2] << std::endl;
        // }

        // unmap texture coordinate buffer object
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::TEX_COORD ) ]->getName() );
        // unmap singularity texture buffer object
        cudaGLUnmapBufferObject( buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->getName() );

        // copy PBO to texture
        buffers_[ static_cast<int>( Buffers::SING_TEX ) ]->bind();
        // glEnable( GL_TEXTURE_2D );
        singTex_->bindTexture();
        glTexSubImage2D( GL_TEXTURE_2D,
                         0,
                         0,
                         0,
                         ( numProbes / params.texSize + 1 ) * this->probeNeighborCount,
                         numProbes % params.texSize,
                         GL_RGB,
                         GL_FLOAT,
                         NULL );
        glBindTexture( GL_TEXTURE_2D, 0 );
        // glDisable( GL_TEXTURE_2D );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    }

    cb::ContourBuildupGraphics ContourBuildup::getGraphics() const
    {
        return { _molecule.size, sphereVAO_, numProbes, sphericalTriangleVAO_, numSC, torusVAO_, singTex_->getName() };
    }

    void ContourBuildup::writeAtomPositionsVBO()
    {
        // Atoms are already in vec4 format
        // write atoms to array
        // for ( unsigned int cnt = 0; cnt < mol->AtomCount(); cnt++ )
        // {
        //     // write pos and rad to array
        //     hPos_[ cnt ] = glm::vec4( glm::make_vec3( &mol->AtomPositions()[ cnt * 3 ] ),
        //                               mol->AtomTypes()[ mol->AtomTypeIndices()[ cnt ] ].Radius() );
        // }

        if ( buffers_[ static_cast<int>( Buffers::ATOM_POS ) ] == nullptr )
        {
            buffers_[ static_cast<int>( Buffers::ATOM_POS ) ] = std::make_unique<glowl::BufferObject>(
                GL_ARRAY_BUFFER, _molecule.ptr, _molecule.size * 4 * sizeof( float ), GL_DYNAMIC_DRAW );
            cudaGLRegisterBufferObject( buffers_[ static_cast<int>( Buffers::ATOM_POS ) ]->getName() );
        }
        else
        {
            buffers_[ static_cast<int>( Buffers::ATOM_POS ) ]->rebuffer( _molecule.ptr,
                                                                         _molecule.size * 4 * sizeof( float ) );
        }

        if ( sphereVAO_ == 0 )
        {
            glGenVertexArrays( 1, &sphereVAO_ );
            glBindVertexArray( sphereVAO_ );

            buffers_[ static_cast<int>( Buffers::ATOM_POS ) ]->bind();
            glEnableVertexAttribArray( 0 );
            glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, nullptr );

            glBindVertexArray( 0 );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
            glDisableVertexAttribArray( 0 );
        }
    }
} // namespace bcs
