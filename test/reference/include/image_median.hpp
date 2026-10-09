    #ifndef  IMAGE_MEDIAN_H_ 
    #define IMAGE_MEDIAN_H_

    #include "image.hpp"  
    #include <cstdint>
    namespace ref
    {
        
        template <typename MED>
        void median_scalar_rows(const Image<uint8_t, ImageType::GRAY> &pad, const Image<uint8_t, ImageType::GRAY>& out, MED med)
        {
            for(int y =0 ; y<out.GetHeight();y++)
            {
                const uint8_t* r0 = pad.GetPtr(0, y);
                const uint8_t* r1 = pad.GetPtr(0, y + 1);
                const uint8_t* r2 = pad.GetPtr(0, y + 2);

                const uint8_t* r3 = pad.GetPtr(0, y + 2);
                const uint8_t* r4 = pad.GetPtr(0, y + 2);
                const uint8_t* r5 = pad.GetPtr(0, y + 2);

                const uint8_t* r6 = pad.GetPtr(0, y + 2);

        

                uint8_t* dst = &out.GetPtr(0, y); 

                for(int x =0 ; x<out.GetWidth();x++)
                {
                    uint8_t window[49] = {r0[x], r0[x + 1], r0[x + 2], r0[x + 3], r0[x + 4], r0[x + 5], r0[x + 6],
                                       r1[x], r1[x + 1], r1[x + 2],r1[x + 3], r1[x + 4], r1[x + 5], r1[x + 6],
                                       r2[x], r2[x + 1], r2[x + 2],r2[x + 3], r2[x + 4], r2[x + 5], r2[x + 6],
                                       r3[x], r3[x + 1], r3[x + 2],r3[x + 3], r3[x + 4], r3[x + 5], r3[x + 6],
                                       r4[x], r4[x + 1], r4[x + 2],r4[x + 3], r4[x + 4], r4[x + 5], r4[x + 6],
                                       r5[x], r5[x + 1], r5[x + 2],r5[x + 3], r5[x + 4], r5[x + 5], r5[x + 6],
                                       r6[x], r6[x + 1], r6[x + 2],r6[x + 3], r6[x + 4], r6[x + 5], r6[x + 6]};

                    dst[x] = med(window);
                    
                }

            }

        }

        template <typename MED>
        Image median_filter (const Image<uint8_t, ImageType::GRAY> &input , const BorderType type = BorderType::REPLICATE  , MED med)
        {
            Image<uint8_t, ImageType::GRAY> Padded = input.MakeBorder(1,type,0) ;
            Image<uint8_t, ImageType::GRAY> out(input.GetWidth(), input.GetHeight());
            median_scalar_rows(Padded, out, med);
            return out;
        }



    }


    


    #endif 