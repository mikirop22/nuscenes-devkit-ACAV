## model_transformer_1.pth

Aquesta versió del model inclou millores significatives en l'entrenament i el preprocessament de dades:

* **Funció de Pèrdua Física:** Incorpora `vel_loss` (velocitat), `acc_loss` (acceleració) i un pes elevat a la `fde_loss` per garantir la coherència física i reduir la deriva (*drift*).
* **Correcció de Coordenades:** S'ha afegit la transformació de vectors al sistema de referència local de l'ego (Global $\to$ Local), alineant correctament els agents amb el mapa BEV.
* **Agent Masking:** Implementació de l'**`agent_mask`** al Transformer. Això permet al model ignorar el *padding* (zeros), evitant "col·lisions fantasma" i millorant dràsticament la convergència.
* **Inputs:** Ara s'utilitzen primer el BEV, també les posicions (x, y), la velocitat (vx, vy) dels agents, l'acceleració (ax, ay), el yaw rate (cos(yaw), sin(yaw)), la width i la length de la bounding box.

OUTPUT METRICS DELS MODES (mini_val):
🔬 Inspeccionando 5 ejemplos del dataset...

================================================================================
🚗 Muestra 0 | Token: c5f58c19249d4137ae063b0e9ecd8b8e
   Ground Truth Final (Local): (4.08, 0.38)
------------------------------------------------------------
   Modo  | Prob     | ADE      | FDE      | Destino Predicho (Local)
------------------------------------------------------------
   0     | 0.0165   | 1.5879   | 1.9182   | (5.43, 1.74)
   1     | 0.0136   | 1.6202   | 3.0443   | (1.05, 0.67)
   2 *   | 0.9699   | 10.5880   | 18.6426   | (16.74, 14.06)
------------------------------------------------------------
   ❌ El modelo eligió el modo 2, pero el mejor era el 0.
      Desperdicio de precisión: 9.00m en ADE
================================================================================

🚗 Muestra 1 | Token: 700c1a25559b4433be532de3475e58a9
   Ground Truth Final (Local): (4.99, 0.53)
------------------------------------------------------------
   Modo  | Prob     | ADE      | FDE      | Destino Predicho (Local)
------------------------------------------------------------
   0     | 0.0152   | 1.1373   | 1.2414   | (5.29, 1.73)
   1     | 0.0137   | 2.0348   | 4.0382   | (0.96, 0.67)
   2 *   | 0.9711   | 10.2479   | 18.1476   | (16.62, 14.47)
------------------------------------------------------------
   ❌ El modelo eligió el modo 2, pero el mejor era el 0.
      Desperdicio de precisión: 9.11m en ADE
================================================================================

🚗 Muestra 2 | Token: 747aa46b9a4641fe90db05d97db2acea
   Ground Truth Final (Local): (6.64, 1.06)
------------------------------------------------------------
   Modo  | Prob     | ADE      | FDE      | Destino Predicho (Local)
------------------------------------------------------------
   0     | 0.0131   | 0.8738   | 1.5089   | (5.30, 1.75)
   1     | 0.0127   | 2.4802   | 5.6795   | (0.98, 0.63)
   2 *   | 0.9742   | 9.7938   | 16.8765   | (16.34, 14.87)
------------------------------------------------------------
   ❌ El modelo eligió el modo 2, pero el mejor era el 0.
      Desperdicio de precisión: 8.92m en ADE
================================================================================

🚗 Muestra 3 | Token: f4f86af4da3b49e79497deda5c5f223a
   Ground Truth Final (Local): (8.16, 1.88)
------------------------------------------------------------
   Modo  | Prob     | ADE      | FDE      | Destino Predicho (Local)
------------------------------------------------------------
   0     | 0.0117   | 0.9436   | 2.7560   | (5.41, 1.77)
   1     | 0.0107   | 3.0553   | 7.2295   | (1.04, 0.64)
   2 *   | 0.9777   | 9.0258   | 14.7244   | (16.07, 14.30)
------------------------------------------------------------
   ❌ El modelo eligió el modo 2, pero el mejor era el 0.
      Desperdicio de precisión: 8.08m en ADE
================================================================================

🚗 Muestra 4 | Token: 6832e717621341568c759151b5974512
   Ground Truth Final (Local): (9.38, 2.98)
------------------------------------------------------------
   Modo  | Prob     | ADE      | FDE      | Destino Predicho (Local)
------------------------------------------------------------
   0     | 0.0190   | 1.2088   | 3.9488   | (5.61, 1.81)
   1     | 0.0159   | 3.6022   | 8.6605   | (1.01, 0.76)
   2 *   | 0.9651   | 8.4672   | 13.7402   | (15.83, 15.11)
------------------------------------------------------------
   ❌ El modelo eligió el modo 2, pero el mejor era el 0.
      Desperdicio de precisión: 7.26m en ADE
================================================================================

## model_transformer_2.pth

Aquesta versió del model inclou les següents millores addicionals, centrades específicament en com es calcula i es penalitza l'error multimodal:

* **Millora en la selecció del "Millor Mode" (Physics-Aware Selection):**
    * *Abans:* Es seleccionava la millor trajectòria basant-se únicament en la distància L2 mitjana (**ADE**). Això podia seleccionar trajectòries que estaven a prop espacialment però tenien velocitats incorrectes.
    * *Ara:* Utilitzem una **mètrica d'error total ponderada** que suma l'error de posició, l'error de velocitat (`vel_err`) i l'error de punt final (`fde_err`). Això assegura que el mode triat com a "correcte" per a l'entrenament sigui cinemàticament coherent.

* **Soft Classification amb Divergència KL:**
    * *Abans:* S'utilitzava *Cross Entropy* amb una assignació dura (*Hard Assignment*). El model només rebia feedback positiu per a un sol mode (el millor) i negatiu per a tots els altres, encara que un segon mode fos gairebé perfecte.
    * *Ara:* Implementem **`F.kl_div`** (Kullback-Leibler Divergence) amb *Soft Targets*. Calculem la distribució objectiu mitjançant un `softmax` negatiu de l'error total (`mode_target = F.softmax(-total_mode_error)`). Això ensenya al model a assignar probabilitats altes a totes les trajectòries plausibles, no només a una, gestionant millor la incertesa multimodal.


## model_transformer_3.pth

Aquesta versió del model inclou les següents millores addicionals:

* **Incorporació dels anchors en les modes del model:** Ara el model té una funció per cada mode basada en els anchors predefinits, millorant la diversitat i precisió de les prediccions multimodals.


## model_transformer_4.pth

Aquesta versió del model inclou les següents millores addicionals:

* **Ús d'anchors dinàmics basats en K-Means per a cada mostra:** En lloc d'utilitzar un conjunt fix d'anchors per a totes les mostres, ara es generen anchors personalitzats per a cada exemple utilitzant K-Means clustering i un ajust d'aquests valors depenent de l'estat actual de l'agent (la velocitat o direcció o distància recorreguda). Això permet al model adaptar-se millor a la variabilitat de les trajectòries reals dels agents.


## model_transformer_5.pth

Aquesta versió del model inclou les següents millores addicionals:

* **Correcció de la imatge BEV per a l'alineació amb els vectors de l'agent:** Ara la imatge BEV s'ha rotat -90 graus per assegurar que l'eix X+ de la imatge coincideixi amb la direcció cap endavant de l'agent. Això millora la coherència entre les dades d'entrada i les trajectòries previstes pel model.


## model_transformer_6.pth

Aquesta versió del model inclou les següents millores addicionals:

* **Criterion amb pes més alt per a la classificació:** S'ha augmentat el pes de la pèrdua de classificació dins del `MultiModalTrajectoryLoss`, donant més importància a la correcta assignació de probabilitats als modes predits pel model. Això ajuda a millorar la precisió global de les prediccions multimodals.


## model_transformer_7.pth

Aquesta versió del model inclou les següents millores addicionals:

* **Añadir més capes semàntiques a la imatge BEV d'entrada:** S'han incorporat capes addicionals que representen elements com carreteres, voreres i senyals de trànsit. Aquesta informació enriquida proporciona al model un context més complet sobre l'entorn, millorant la seva capacitat per predir trajectòries realistes i segures.