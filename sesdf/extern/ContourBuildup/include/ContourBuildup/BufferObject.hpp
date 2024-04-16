/*
 * BufferObject.hpp
 *
 * MIT License
 * Copyright (c) 2019 Michael Becher
 */

#ifndef GLOWL_BUFFEROBJECT_HPP
#define GLOWL_BUFFEROBJECT_HPP

#include <stdexcept>
#include <string>

#include <GL/gl3w.h>

namespace glowl
{

	/**
	 * \class BufferObject
	 *
	 * \brief Generic OpenGL buffer object.
	 *
	 * \author Michael Becher
	 */
	class BufferObject
	{
	  public:
		/**
		 * \brief BufferObject constructor that uses std containers as input.
		 *
		 * Note: Active OpenGL context required for construction.
		 * Use std::unqiue_ptr (or shared_ptr) for delayed construction of class member variables of this type.
		 */
		template<typename Container>
		BufferObject( GLenum target, Container const & datastorage, GLenum usage = GL_DYNAMIC_DRAW );

		/**
		 * \brief Bufferobject constructor that uses data pointer and byte size as input.
		 *
		 * Note: Active OpenGL context required for construction.
		 * Use std::unqiue_ptr (or shared_ptr) for delayed construction of class member variables of this type.
		 */
		BufferObject( GLenum target, GLvoid const * data, GLsizeiptr byte_size, GLenum usage = GL_DYNAMIC_DRAW );

		~BufferObject();

		BufferObject( const BufferObject & cpy )			 = delete;
		BufferObject( BufferObject && other )				 = delete;
		BufferObject & operator=( BufferObject && rhs )		 = delete;
		BufferObject & operator=( const BufferObject & rhs ) = delete;

		template<typename Container>
		void bufferSubData( Container const & datastorage, GLsizeiptr byte_offset = 0 ) const;

		void bufferSubData( GLvoid const * data, GLsizeiptr byte_size, GLsizeiptr byte_offset = 0 ) const;

		template<typename Container>
		void rebuffer( Container const & datastorage );

		void rebuffer( GLvoid const * data, GLsizeiptr byte_size );

		void bind() const;

		void bind( GLuint index ) const;

		void bindAs( GLenum target, GLuint index ) const;

		static void copy( BufferObject * src, BufferObject * tgt );

		static void copy( BufferObject * src,
						  BufferObject * tgt,
						  GLintptr		 readOffset,
						  GLintptr		 writeOffset,
						  GLsizeiptr	 size );

		GLenum getTarget() const;

		GLuint getName() const;

		GLsizeiptr getByteSize() const;

	  private:
		GLenum	   m_target;
		GLuint	   m_name;
		GLsizeiptr m_byte_size;
		GLenum	   m_usage;
	};

	template<typename Container>
	inline BufferObject::BufferObject( GLenum target, Container const & datastorage, GLenum usage ) :
		m_target( target ), m_name( 0 ),
		m_byte_size( static_cast<GLsizeiptr>( datastorage.size() * sizeof( typename Container::value_type ) ) ),
		m_usage( usage )
	{
		glCreateBuffers( 1, &m_name );
		glNamedBufferData( m_name, m_byte_size, datastorage.data(), m_usage );

		auto err = glGetError();
		if ( err != GL_NO_ERROR )
		{
			throw std::runtime_error( "BufferObject::BufferObject - OpenGL error " + std::to_string( err ) );
		}
	}

	inline BufferObject::BufferObject( GLenum target, GLvoid const * data, GLsizeiptr byte_size, GLenum usage ) :
		m_target( target ), m_name( 0 ), m_byte_size( byte_size ), m_usage( usage )
	{
		glCreateBuffers( 1, &m_name );
		glNamedBufferData( m_name, m_byte_size, data, m_usage );

		auto err = glGetError();
		if ( err != GL_NO_ERROR )
		{
			throw std::runtime_error( "BufferObject::BufferObject - OpenGL error " + std::to_string( err ) );
		}
	}

	inline BufferObject::~BufferObject() { glDeleteBuffers( 1, &m_name ); }

	template<typename Container>
	inline void BufferObject::bufferSubData( Container const & datastorage, GLsizeiptr byte_offset ) const
	{
		// check if feasible
		if ( ( byte_offset + static_cast<GLsizeiptr>( datastorage.size() * sizeof( typename Container::value_type ) ) )
			 > m_byte_size )
		{
			// error message
			throw std::runtime_error( "BufferObject::bufferSubData - given data too large for buffer" );
		}

		glNamedBufferSubData(
			m_name, byte_offset, datastorage.size() * sizeof( typename Container::value_type ), datastorage.data() );
	}

	inline void BufferObject::bufferSubData( GLvoid const * data, GLsizeiptr byte_size, GLsizeiptr byte_offset ) const
	{
		// check if feasible
		if ( ( byte_offset + byte_size ) > m_byte_size )
		{
			// error message
			throw std::runtime_error( "BufferObject::bufferSubData - given data too large for buffer" );
		}

		glNamedBufferSubData( m_name, byte_offset, byte_size, data );
	}

	template<typename Container>
	inline void BufferObject::rebuffer( Container const & datastorage )
	{
		m_byte_size = static_cast<GLsizeiptr>( datastorage.size() * sizeof( typename Container::value_type ) );
		glNamedBufferData( m_name, m_byte_size, datastorage.data(), m_usage );

		auto err = glGetError();
		if ( err != GL_NO_ERROR )
		{
			throw std::runtime_error( "BufferObject::rebuffer - OpenGL error " + std::to_string( err ) );
		}
	}

	inline void BufferObject::rebuffer( GLvoid const * data, GLsizeiptr byte_size )
	{
		m_byte_size = byte_size;
		glNamedBufferData( m_name, m_byte_size, data, m_usage );

		auto err = glGetError();
		if ( err != GL_NO_ERROR )
		{
			throw std::runtime_error( "BufferObject::rebuffer - OpenGL error " + std::to_string( err ) );
		}
	}

	inline void BufferObject::bind() const { glBindBuffer( m_target, m_name ); }

	inline void BufferObject::bind( GLuint index ) const { glBindBufferBase( m_target, index, m_name ); }

	inline void BufferObject::bindAs( GLenum target, GLuint index ) const
	{
		glBindBufferBase( target, index, m_name );
		auto err = glGetError();
		if ( err != GL_NO_ERROR )
		{
			throw std::runtime_error( "BufferObject::bindAs - OpenGL error " + std::to_string( err ) );
		}
	}

	inline void BufferObject::copy( BufferObject * src, BufferObject * tgt )
	{
		if ( src->m_byte_size > tgt->m_byte_size )
		{
			throw std::runtime_error( "BufferObject::copy - target buffer smaller than source" );
		}

		glCopyNamedBufferSubData( src->m_name, tgt->m_name, 0, 0, src->m_byte_size );
	}

	inline void BufferObject::copy( BufferObject * src,
									BufferObject * tgt,
									GLintptr	   readOffset,
									GLintptr	   writeOffset,
									GLsizeiptr	   size )
	{
		if ( ( readOffset + size ) > src->m_byte_size )
		{
			throw std::runtime_error( "BufferObject::copy - source buffer out of bounds" );
		}
		else if ( ( writeOffset + size ) > tgt->m_byte_size )
		{
			throw std::runtime_error( "BufferObject::copy - target buffer out of bounds" );
		}

		glCopyNamedBufferSubData( src->m_name, tgt->m_name, readOffset, writeOffset, size );
	}

	inline GLenum BufferObject::getTarget() const { return m_target; }

	inline GLuint BufferObject::getName() const { return m_name; }

	inline GLsizeiptr BufferObject::getByteSize() const { return m_byte_size; }

} // namespace glowl

#endif // GLOWL_BUFFEROBJECT_HPP
