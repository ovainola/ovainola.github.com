#include "class_example.h"

using namespace vehicles;

Car::Car(char* brand, char* model, int yearBought, double currentSpeed)
{
    cBrand = brand;
    cModel = model;
    cYear  = yearBought;
    mSpeed = currentSpeed;
}

Car::~Car()
{
}

double Car::getSpeed() {
    return mSpeed;
}

void Car::setSpeed(double newSpeed) {
  mSpeed = newSpeed;
}
