#ifndef BCS_CONTOURBUILDUP_GRAPHICS_HPP
#define BCS_CONTOURBUILDUP_GRAPHICS_HPP

using GLuint = unsigned int;
namespace bcs::cb
{
    struct ContourBuildupGraphics
    {
        std::size_t atomNb;
        GLuint      atomVao;

        std::size_t sphericalTriangleNb;
        GLuint      sphericalTriangleVao;

        std::size_t torusNb;
        GLuint      torusVao;

        GLuint singularityTexture;
    };
} // namespace bcs::cb

#endif // BCS_CONTOURBUILDUP_GRAPHICS_HPP
