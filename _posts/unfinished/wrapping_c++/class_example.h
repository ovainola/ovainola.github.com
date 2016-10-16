#pragma once

namespace vehicles {
  class Car {
    private:
      const char* cBrand;
      const char* cModel;
      int cYear;
      double mSpeed;
    public:
      Car(char* brand, char* model, int yearBought, double currentSpeed);
      ~Car();
      double getSpeed();
      void setSpeed(double newSpeed);
  };
}
