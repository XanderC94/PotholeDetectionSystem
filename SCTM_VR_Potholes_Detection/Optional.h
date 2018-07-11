//
// Created by Xander_C on 25/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_OPTIONAL_H
#define POTHOLEDETECTIONSYSTEM_OPTIONAL_H

namespace cv {
    template<class T>
    class Optional {

    private:
        T value;
        bool empty;
    public:

        Optional(T value) {
            this->value = value;
            this->empty = false;
        }

        Optional() {
            this->empty = true;
        }

        ~Optional() {}

        T getValue() {
            return this->value;
        }

        bool hasValue() {
            return !this->empty;
        }
    };
}


#endif //POTHOLEDETECTIONSYSTEM_OPTIONAL_H
