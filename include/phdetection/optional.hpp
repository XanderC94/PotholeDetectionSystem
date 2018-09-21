//
// Created by Xander_C on 25/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_OPTIONAL_H
#define POTHOLEDETECTIONSYSTEM_OPTIONAL_H

namespace cv {
    template<class T>
    class optional {

    private:
        T value;
        bool empty;
    public:

        optional(T value) {
            this->value = value;
            this->empty = false;
        }

        optional() {
            this->empty = true;
        }

        ~optional() {}

        T getValue() {
            return this->value;
        }

        bool hasValue() {
            return !this->empty;
        }
    };
}


#endif //POTHOLEDETECTIONSYSTEM_OPTIONAL_H
