/*
 * Texture.hpp
 *
 * MIT License
 * Copyright (c) 2019 Michael Becher
 */

#ifndef GLOWL_TEXTURE_HPP
#define GLOWL_TEXTURE_HPP

#include <string>
#include <vector>

#include <GL/gl3w.h>

namespace glowl
{

	struct TextureLayout
	{
		TextureLayout() : internal_format( 0 ), width( 0 ), height( 0 ), depth( 0 ), format( 0 ), type( 0 ), levels( 0 )
		{
		}
		/**
		 * \param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
		 * \param width Specifies the width of the texture in pixels.
		 * \param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
		 * \param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
		 * \param format Specifies the format of the texture (e.g. GL_RGBA)
		 * \param type Specifies the type of the texture (e.g. GL_FLOAT)
		 */
		TextureLayout( GLint   internal_format,
					   int	   width,
					   int	   height,
					   int	   depth,
					   GLenum  format,
					   GLenum  type,
					   GLsizei levels ) :
			internal_format( internal_format ),
			width( width ), height( height ), depth( depth ), format( format ), type( type ), levels( levels )
		{
		}

		/**
		 * \param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
		 * \param width Specifies the width of the texture in pixels.
		 * \param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
		 * \param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
		 * \param format Specifies the format of the texture (e.g. GL_RGBA)
		 * \param type Specifies the type of the texture (e.g. GL_FLOAT)
		 * \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
		 * {{GL_TEXTURE_SPARSE_ARB,GL_TRUE},{...},...} \param int_parameters A list of float texture parameters, each
		 * given by a pair of name and value (e.g. {{GL_TEXTURE_MAX_ANISOTROPY_EX,4.0f},{...},...}
		 */
		TextureLayout( GLint										   internal_format,
					   int											   width,
					   int											   height,
					   int											   depth,
					   GLenum										   format,
					   GLenum										   type,
					   GLsizei										   levels,
					   std::vector<std::pair<GLenum, GLint>> const &   int_parameters,
					   std::vector<std::pair<GLenum, GLfloat>> const & float_parameters ) :
			internal_format( internal_format ),
			width( width ), height( height ), depth( depth ), format( format ), type( type ), levels( levels ),
			int_parameters( int_parameters ), float_parameters( float_parameters )
		{
		}
		TextureLayout( GLint									  internal_format,
					   int										  width,
					   int										  height,
					   int										  depth,
					   GLenum									  format,
					   GLenum									  type,
					   GLsizei									  levels,
					   std::vector<std::pair<GLenum, GLint>> &&	  int_parameters,
					   std::vector<std::pair<GLenum, GLfloat>> && float_parameters ) :
			internal_format( internal_format ),
			width( width ), height( height ), depth( depth ), format( format ), type( type ), levels( levels ),
			int_parameters( int_parameters ), float_parameters( float_parameters )
		{
		}

		GLint  internal_format;
		int	   width;
		int	   height;
		int	   depth;
		GLenum format;
		GLenum type;

		GLsizei levels;

		std::vector<std::pair<GLenum, GLint>>	int_parameters;
		std::vector<std::pair<GLenum, GLfloat>> float_parameters;
	};

	/**
	 * \class Texture
	 *
	 * \brief Abstract base class for various OpenGL texture Objects (2D,3D,2DArray).
	 *
	 * \author Michael Becher
	 */
	class Texture
	{
	  protected:
		std::string m_id; ///< Identifier set by application to help identifying textures

		GLuint m_name; ///< OpenGL texture name given by glCreateTexture
#ifndef GLOWL_NO_ARB_BINDLESS_TEXTURE
		GLuint64 m_texture_handle; ///< Actual OpenGL texture handle (used for bindless)
#endif

		GLenum m_internal_format;
		GLenum m_format;
		GLenum m_type;

		GLsizei m_levels;

		// TODO: Store texture parameters as well ?
	  public:
		Texture( std::string id, GLint internal_format, GLenum format, GLenum type, GLsizei levels ) :
			m_id( id ), m_internal_format( internal_format ), m_format( format ), m_type( type ), m_levels( levels )
		{
		}
		virtual ~Texture() {}
		Texture( const Texture & ) = delete;

		virtual void bindTexture() const = 0;

		// TODO: Deprecate simplified function in the future
		void bindImage( GLuint location, GLenum access ) const
		{
			glBindImageTexture( location, m_name, 0, GL_TRUE, 0, access, m_internal_format );
		}

		void bindImage( GLuint location, GLint level, GLboolean layered, GLint layer, GLenum access ) const
		{
			glBindImageTexture( location, m_name, level, layered, layer, access, m_internal_format );
		}

		virtual void updateMipmaps() = 0;

		virtual TextureLayout getTextureLayout() const = 0;

		std::string getId() const { return m_id; }

		GLuint getName() const { return m_name; }

		GLenum getInternalFormat() const { return m_internal_format; }
		GLenum getFormat() const { return m_format; }
		GLenum getType() const { return m_type; }
	};

} // namespace glowl

#endif // GLOWL_TEXTURE_HPP
