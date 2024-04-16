#ifndef BCS_CONTOURBUILDUP_CONTOURBUILDUP_HPP
#define BCS_CONTOURBUILDUP_CONTOURBUILDUP_HPP

#include <array>
#include <memory>
#include <vector>

#include <ContourBuildup/particles_kernel.cuh>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "bcs/contour_buildup/graphics.hpp"
#include "bcs/core/molecule.hpp"
#include "bcs/core/type.hpp"

using uint = unsigned int;
namespace glowl
{
    class BufferObject;
    class Texture2D;
} // namespace glowl

namespace bcs
{
    class ContourBuildup
    {
      public:
        ContourBuildup( const ConstSpan<Vec4f> molecule,
                        const Aabb &           aabb,
                        const float            probeRadius    = 1.4f,
                        uint32_t               gridDimensions = 16,
                        bool                   buildSurface   = true );

        ContourBuildup( const ContourBuildup & other )             = delete;
        ContourBuildup & operator=( const ContourBuildup & other ) = delete;

        ContourBuildup( ContourBuildup && other ) noexcept;
        ContourBuildup & operator=( ContourBuildup && other ) noexcept;

        ~ContourBuildup();

        void                                          build();
        [[nodiscard]] bcs::cb::ContourBuildupGraphics getGraphics() const;

      private:
        void writeAtomPositionsVBO();

        ConstSpan<Vec4f> _molecule;

        GLuint sphereVAO_            = 0;
        GLuint torusVAO_             = 0;
        GLuint sphericalTriangleVAO_ = 0;

        // radius of the probe atom
        float probeRadius;

        // max number of neighbors per atom
        unsigned int atomNeighborCount;

        // params
        bool       cudaInitalized;
        uint       numAtoms;
        SimParams  params;
        glm::uvec3 gridSize;
        uint       numGridCells;

        // CPU data
        // std::vector<glm::vec4> hPos_;

        // GPU data
        float * m_dPos;
        float * m_dSortedPos;
        float * m_dSortedProbePos;
        uint *  m_dNeighborCount;
        uint *  m_dNeighbors;
        float * m_dSmallCircles;
        uint *  m_dSmallCircleVisible;
        uint *  m_dSmallCircleVisibleScan;
        float * m_dArcs;
        uint *  m_dArcIdxK;
        uint *  m_dArcCount;
        uint *  m_dArcCountScan;

        // grid data for sorting method
        uint * m_dGridParticleHash;  // grid hash value for each particle
        uint * m_dGridParticleIndex; // particle index for each particle
        uint * m_dGridProbeHash;     // grid hash value for each probe
        uint * m_dGridProbeIndex;    // particle index for each probe
        uint * m_dCellStart;         // index of start of each cell in sorted list
        uint * m_dCellEnd;           // index of end of cell
        uint   gridSortBits;
        uint   numProbes;
        uint   numSC;

        enum class Buffers : GLuint
        {
            PROBE_POS         = 0,
            SPHERE_TRIA_VEC_1 = 1,
            SPHERE_TRIA_VEC_2 = 2,
            SPHERE_TRIA_VEC_3 = 3,
            TORUS_POS         = 4,
            TORUS_VS          = 5,
            TORUS_AXIS        = 6,
            SING_TEX          = 7,
            TEX_COORD         = 8,
            ATOM_POS          = 9,
            BUFF_COUNT        = 10
        };
        std::array<std::unique_ptr<glowl::BufferObject>, static_cast<int>( Buffers::BUFF_COUNT )> buffers_;

        // singularity texture
        std::unique_ptr<glowl::Texture2D> singTex_;

        // maximum number of probe neighbors
        uint         probeNeighborCount;
        unsigned int texHeight;
        unsigned int texWidth;
        unsigned int width;
        unsigned int height;

        bool setCUDAGLDevice;
    };
} // namespace bcs

#endif // BCS_CONTOURBUILDUP_CONTOURBUILDUP_HPP
