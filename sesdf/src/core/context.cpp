#include "bcs/core/context.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <GL/gl3w.h>
#include <fmt/printf.h>

namespace bcs
{
    Context::Context()
    {
#ifndef NDEBUG
        fmt::print( "Initializing SDL2\n" );
#endif // NDEBUG

        if ( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER ) != 0 )
        {
            throw std::runtime_error( SDL_GetError() );
        }

        SDL_GL_SetAttribute( SDL_GL_CONTEXT_FLAGS, 0 );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 4 );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 5 );
        SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
        SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 24 );
        SDL_GL_SetAttribute( SDL_GL_STENCIL_SIZE, 8 );

        SDL_GetCurrentDisplayMode( 0, &m_displayMode );

        m_window = SDL_CreateWindow(
            "Benchmark CB v SESDF",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            1,
            1,
            SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI );

        if ( m_window == nullptr )
            throw std::runtime_error( SDL_GetError() );

        m_glContext = SDL_GL_CreateContext( m_window );
        if ( m_glContext == nullptr )
            throw std::runtime_error( SDL_GetError() );

        SDL_GL_MakeCurrent( m_window, m_glContext );

#ifndef NDEBUG
        fmt::print( "Initializing OpenGL\n" );
#endif // NDEBUG
        if ( gl3wInit() )
            throw std::runtime_error( "gl3wInit() failed" );

        if ( !gl3wIsSupported( 4, 5 ) )
            throw std::runtime_error( "OpenGL version not supported" );
    }

    Context::Context( Context && other ) noexcept
    {
        std::swap( m_window, other.m_window );
        std::swap( m_glContext, other.m_glContext );
        std::swap( m_displayMode, other.m_displayMode );
        std::swap( m_showDialogImport, other.m_showDialogImport );
        std::swap( m_isVisible, other.m_isVisible );
    }

    Context & Context::operator=( Context && other ) noexcept
    {
        std::swap( m_window, other.m_window );
        std::swap( m_glContext, other.m_glContext );
        std::swap( m_displayMode, other.m_displayMode );
        std::swap( m_showDialogImport, other.m_showDialogImport );
        std::swap( m_isVisible, other.m_isVisible );

        return *this;
    }

    Context::~Context()
    {
        if ( m_glContext )
            SDL_GL_DeleteContext( m_glContext );

        if ( m_window )
            SDL_DestroyWindow( m_window );

        SDL_Quit();
    }
} // namespace bcs
