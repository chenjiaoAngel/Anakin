/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SABER_LITE_FUNCS_TIMER_LITE_H
#define ANAKIN_SABER_LITE_FUNCS_TIMER_LITE_H

#include "saber/lite/core/common_lite.h"
#include <chrono>
#include <list>

namespace anakin{

namespace saber{

namespace lite{

//template <typename TargetType>
class SaberTimer final {

public:
    SaberTimer() {}

    ~SaberTimer() {}

    void clear() {
        ms_time.clear();
    }

    void start() {
        tstart = std::chrono::system_clock::now();
    }

    void end() {
        tend = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart);
        float elapse_ms = 1000.f * float(ts.count()) * std::chrono::microseconds::period::num / \
            std::chrono::microseconds::period::den;
        ms_time.push_back(elapse_ms);
    }

    float get_average_ms() {
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum / ms_time.size();
    }

    // return tile (0-99) time.
    float get_tile_time(float tile) {

        if (tile <0 || tile > 100) {
            return -1.f;
        }
        int total_items = (int)ms_time.size();
        if (total_items <= 0) {
            return -2.f;
        }
        ms_time.sort();
        int pos = (int)(tile * total_items / 100);
        auto it = ms_time.begin();
        for (int i = 0; i < pos; ++i) {
            ++it;
        }
        return *it;
    }

    const std::list<float> get_time_stat() {
        return ms_time;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> tstart;
    std::chrono::time_point<std::chrono::system_clock> tend;
    std::list<float> ms_time;
};

} //namespace lite

}  //namespace saber

}  //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_TIMER_LITE_H
