using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FreeWorld : MonoBehaviour
{
    public Camera woldCamera;
    public Camera dashCamera;
    public GameObject viewCamera;
    public GameObject dashCameraGameObject;


    private MonoBehaviour carCrontroller;
    private FreeCamera freeCamera;
 
    private bool usarFreeCamera = false;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    public Vector3 initialFreePosition;
    public Quaternion initialFreeRotation;
    private GameObject cameraGameObject;
    private SmoothCameraFollow smoothCameraFollow;



    void Start()
    {
       
       
            carCrontroller = GetComponent<CarInputController>();
            freeCamera = viewCamera.GetComponent<FreeCamera>();
            
            cameraGameObject = woldCamera.gameObject;
            smoothCameraFollow = cameraGameObject.GetComponent<SmoothCameraFollow>();
         

            if (carCrontroller == null || freeCamera == null)
            {
                Debug.LogError("Os scripts CarInputController e FreeCamera são obrigatórios.");
                return;
            }

            initialPosition = woldCamera.transform.position;
            initialRotation = woldCamera.transform.rotation;
        
            initialFreePosition = dashCameraGameObject.transform.localPosition;
            initialFreeRotation = dashCameraGameObject.transform.localRotation;

            //initialFreePosition  = gameObject.transform.InverseTransformPoint(dashCameraGameObject.transform.position);
            //initialFreeRotation = Quaternion.Inverse(transform.rotation) * dashCameraGameObject.transform.rotation;

           

   
         

        AtualizarModos();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {

            if (usarFreeCamera)
            {
                dashCameraGameObject.transform.localRotation= initialFreeRotation;
                dashCameraGameObject.transform.localPosition = initialFreePosition;
                
                  //  dashCameraGameObject.transform.position = gameObject.transform.TransformPoint(initialFreePosition);
                 //   dashCameraGameObject.transform.rotation = gameObject.transform.rotation * initialFreeRotation;
            }   
        
            usarFreeCamera = !usarFreeCamera;
            AtualizarModos();
        }
        if (usarFreeCamera)
        {
            cameraGameObject.transform.position = freeCamera.transform.position;
            cameraGameObject.transform.rotation = freeCamera.transform.rotation;
        }

    }

    void AtualizarModos()
    {
        if (carCrontroller != null && freeCamera != null)
        {
          
          

                if (usarFreeCamera)
                {
                    cameraGameObject.transform.position = freeCamera.transform.position;
                    cameraGameObject.transform.rotation = freeCamera.transform.rotation;
                }
                else
                {
                    woldCamera.transform.position = initialPosition;
                    woldCamera.transform.rotation = initialRotation;
                }
            smoothCameraFollow.enabled = !usarFreeCamera;
            carCrontroller.enabled = !usarFreeCamera;
            freeCamera.enabled = usarFreeCamera;

            Cursor.lockState = usarFreeCamera ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !usarFreeCamera ? true : false;
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
