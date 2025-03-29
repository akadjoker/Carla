using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine;

public class FreeCamera : MonoBehaviour
{
    public float movementSpeed = 10f;
    public float lookSpeed = 2f;
    public float boostMultiplier = 2f;

    private float rotationX = 0f;
    private float rotationY = 0f;

    void Start()
    {
       
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        // Movimento do rato
        rotationX += Input.GetAxis("Mouse X") * lookSpeed;
        rotationY -= Input.GetAxis("Mouse Y") * lookSpeed;
        rotationY = Mathf.Clamp(rotationY, -90f, 90f);

        transform.localRotation = Quaternion.Euler(rotationY, rotationX, 0);

        // Movimento com teclado
        float speed = Input.GetKey(KeyCode.LeftShift) ? movementSpeed * boostMultiplier : movementSpeed;

        Vector3 move = new Vector3(
            Input.GetAxis("Horizontal"),                // A/D
            (Input.GetKey(KeyCode.E) ? 1 : 0) - (Input.GetKey(KeyCode.Q) ? 1 : 0), // Q/E para cima/baixo
            Input.GetAxis("Vertical")                   // W/S
        );

        transform.Translate(move * speed * Time.deltaTime);
        
        // Soltar o cursor com Esc
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }
}

