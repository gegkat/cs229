using System;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;

namespace UnityStandardAssets.Vehicles.Car
{
    [RequireComponent(typeof(CarController))]
    public class CarRemoteControl : MonoBehaviour
    {
        private CarController m_Car; // the car controller we want to use

        public float SteeringAngle { get; set; }
        public float Acceleration { get; set; }
        private Steering s;

        private void Awake()
        {
            // get the car controller
            m_Car = GetComponent<CarController>();
            s = new Steering();
            s.Start();
        }

        private void FixedUpdate()
        {
            // If holding down W or S control the car manually
            if (Input.GetKey(KeyCode.M))
            {
                s.UpdateValues();

                if (Input.GetKey(KeyCode.N))
                {
                // m_Car.Move(s.H, s.V, s.V, 0f);
                m_Car.Move(s.H, Acceleration, Acceleration, 0f);
                } else if (Input.GetKey(KeyCode.L))
                {
                    // m_Car.Move(s.H, s.V, s.V, 0f);
                    m_Car.Move(SteeringAngle, s.V, s.V, 0f);
                } else
                {
                    m_Car.Move(s.H, s.V, s.V, 0f);
                }

            } else
            {
				m_Car.Move(SteeringAngle, Acceleration, Acceleration, 0f);
            }
        }
    }
}
