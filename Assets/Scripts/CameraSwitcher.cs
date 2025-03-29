using System.Collections;
using System.Collections.Generic;
using UnityEngine;

 

public class CameraSwitcher : MonoBehaviour
{
      private MonoBehaviour followScript;
    private MonoBehaviour freeScript;

    private bool usarFreeCamera = false;

    void Start()
    {
        // Obtemos os scripts da Main Camera
        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            followScript = mainCam.GetComponent<SmoothCameraFollow>();
            freeScript = mainCam.GetComponent<FreeCamera>();
        }

        AtualizarModos();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            usarFreeCamera = !usarFreeCamera;
            AtualizarModos();
        }
    }

    void AtualizarModos()
    {
        if (followScript != null && freeScript != null)
        {
            followScript.enabled = !usarFreeCamera;
            freeScript.enabled = usarFreeCamera;

            Cursor.lockState = usarFreeCamera ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !usarFreeCamera ? true : false;
        }
    }
}
