# distutils: language = c++

cdef extern from "class_example.h" namespace "vehicles":
    cdef cppclass Car:
        Car(char* brand, char* model, int yearBought, double currentSpeed)
        double getSpeed()
        void setSpeed(double newSpeed)

cdef class PyCar:
    cdef Car * pCar
    
    def __cinit__(self, char* pyBrand, char* pyModel, int pyYear, double pySpeed):
        self.pCar = new Car(pyBrand, pyModel, pyYear, pySpeed)

    def get_speed(self):
        return self.pCar.getSpeed()

    def set_car_speed(self, double new_speed):
        self.pCar.setSpeed(new_speed)
