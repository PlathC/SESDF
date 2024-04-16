#ifndef BCS_CORE_CONTEXT_HPP
#define BCS_CORE_CONTEXT_HPP

#include <SDL.h>

namespace bcs
{
    class Context
    {
      public:
        Context();

        Context( const Context & )             = delete;
        Context & operator=( const Context & ) = delete;

        Context( Context && ) noexcept;
        Context & operator=( Context && ) noexcept;

        ~Context();

      private:
        SDL_Window *    m_window    = nullptr;
        SDL_GLContext   m_glContext = nullptr;
        SDL_DisplayMode m_displayMode;
        bool            m_showDialogImport = false;
        bool            m_isVisible        = true;
    };
} // namespace bcs

#endif // BCS_CORE_CONTEXT_HPP
