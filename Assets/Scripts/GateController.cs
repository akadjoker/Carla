using System.Collections;
using System.Collections.Generic;
using UnityEngine;



using UnityEngine;

public class GateController : MonoBehaviour
{
    public bool abrirParaDireita = true;
    public float anguloAbertura = 90f;
    public float velocidade = 45f;
    public float tempoAberta = 3f;  // Tempo que a cancela fica aberta

    private float anguloAtual = 0f;
    private bool aAbrir = false;
    private bool aFechar = false;
    private Quaternion rotacaoInicial;
    private float tempoDesdeAberta = 0f;

    void Start()
    {
        rotacaoInicial = transform.rotation;
    }

    void Update()
    {
        // Tecla para abrir/fechar
        if (Input.GetKeyDown(KeyCode.O) && !aAbrir && !aFechar && anguloAtual == 0f)
        {
            aAbrir = true;
        }
 
        if (aAbrir && Mathf.Abs(anguloAtual) < Mathf.Abs(anguloAbertura))
        {
            float direcao = abrirParaDireita ? 1f : -1f;
            float rotacaoFrame = velocidade * Time.deltaTime * direcao;

            if (Mathf.Abs(anguloAtual + rotacaoFrame) > Mathf.Abs(anguloAbertura))
                rotacaoFrame = anguloAbertura - anguloAtual;

            transform.Rotate(Vector3.left, rotacaoFrame);
            anguloAtual += Mathf.Abs(rotacaoFrame);

            if (Mathf.Abs(anguloAtual) >= Mathf.Abs(anguloAbertura))
            {
                aAbrir = false;
                tempoDesdeAberta = 0f;
            }
        }
 
        if (!aAbrir && anguloAtual >= anguloAbertura)
        {
            tempoDesdeAberta += Time.deltaTime;

            if (tempoDesdeAberta >= tempoAberta)
                aFechar = true;
        }

 
        if (aFechar && anguloAtual > 0f)
        {
            float direcao = abrirParaDireita ? -1f : 1f;
            float rotacaoFrame = velocidade * Time.deltaTime * direcao;

            if (anguloAtual - Mathf.Abs(rotacaoFrame) < 0f)
                rotacaoFrame = anguloAtual * direcao;

            transform.Rotate(Vector3.left, rotacaoFrame);
            anguloAtual -= Mathf.Abs(rotacaoFrame);

            if (anguloAtual <= 0f)
            {
                aFechar = false;
                anguloAtual = 0f;
            }
        }
    }
}

