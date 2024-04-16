#ifndef TIMER_CUH
#define TIMER_CUH

namespace bcs
{
    class TimingEvent
    {
      public:
        TimingEvent()
        {
            cudaEventCreate( &m_start );
            cudaEventCreate( &m_stop );
            m_initialized = true;
        }
        TimingEvent( const TimingEvent & ) = delete;
        TimingEvent & operator=( const TimingEvent & ) = delete;
        TimingEvent( TimingEvent && other )
        {
            std::swap( m_start, other.m_start );
            std::swap( m_stop, other.m_stop );
            std::swap( m_initialized, other.m_initialized );
        }
        TimingEvent & operator=( TimingEvent && other )
        {
            std::swap( m_start, other.m_start );
            std::swap( m_stop, other.m_stop );
            std::swap( m_initialized, other.m_initialized );
        }
        ~TimingEvent()
        {
            if ( m_initialized )
            {
                cudaEventDestroy( m_start );
                cudaEventDestroy( m_stop );
            }
        }

        void start() { cudaEventRecord( m_start ); }
        void stop() { cudaEventRecord( m_stop ); }

        float getElapsedMs()
        {
            cudaEventSynchronize( m_stop );
            float milliseconds = 0;
            cudaEventElapsedTime( &milliseconds, m_start, m_stop );
            return milliseconds;
        }

      private:
        bool        m_initialized = false;
        cudaEvent_t m_start, m_stop;
    };
} // namespace bcs

#endif // TIMER_CUH