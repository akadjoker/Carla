using UnityEngine;

public class SwitchMainCamera : MonoBehaviour
{
    public Camera cameraA;
    public Camera cameraB;

    private bool usarCameraB = false;

    void Start()
    {
        AtivarCamera(cameraA, true);
        AtivarCamera(cameraB, false);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            usarCameraB = !usarCameraB;

            if (usarCameraB)
            {
                AtivarCamera(cameraA, false);
                AtivarCamera(cameraB, true);
            }
            else
            {
                AtivarCamera(cameraA, true);
                AtivarCamera(cameraB, false);
            }
        }
    }

    void AtivarCamera(Camera cam, bool ativa)
    {
        if (cam == null) return;

        cam.enabled = ativa;
        cam.gameObject.SetActive(ativa);
        cam.tag = ativa ? "MainCamera" : "Untagged";
    }
}
