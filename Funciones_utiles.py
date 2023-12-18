import numpy as np 
import matplotlib.pyplot as plt
import pyvista as pv

def Guardar_Gif_Malla(malla, u_arr,name= 'gif_malla'):
    """
    Funcion creada para guardar Gif

    u_arr: Considera las siguientes dimensiones (nodos, dimensiones, tiempo)
    """

    mesh = malla.copy()
    COO = mesh.points.copy()
    pts = mesh.points.copy()

        
    # Create a plotter object and set the scalars to the Z height
    plotter = pv.Plotter(notebook=True, off_screen=True)#(notebook=True, off_screen=True)
    plotter.add_mesh(
        mesh,
        scalars=np.linalg.norm(u_arr[:,:,0],axis = 1),
        lighting=True,
        show_edges=False,cmap= 'inferno',
        scalar_bar_args={"title": "Magnitud Desplazamiento"}
    )

    plotter.set_background('gray')

    plotter.camera_position = 'xy'



    # Open a gif
    texto = name + '.gif'
    plotter.open_gif(texto)

    for i in range(u_arr.shape[-1]):
        COO[:,:2] = pts[:,:2] + u_arr[:,:,i]
        mag = np.linalg.norm(u_arr[:,:,i],axis = 1)
        plotter.update_coordinates(COO, render=False)
        plotter.update_scalars(mag, render=False)

        # Write a frame. This triggers a render.
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()
    return


def malla_deformada(malla,disp, Dim = '2D'):
    """
    Funcion para ver una malla deformada
    """

    mesh = malla.copy()
    mesh.points[:,:2] += disp
    mesh.plot()
    return


def DATOS_TO_imagen(datos, size = (151,51), dim = 2):
    """
    Se considera que los datos entran normalizados con valores en 0 y 1

    La funcion no es del todo general y considera datos de desplzamiento 2D
    """

    normalized_data = datos.reshape((size[0],size[1],dim)) 
    image = np.zeros((size[0],size[1], 3), dtype=np.float64)

    #image[:, :, 0] = (normalized_data[:, :, 0] * 255).astype(np.float64)
    #image[:, :, 1] = (normalized_data[:, :, 1] * 255).astype(np.float64)
    image[:, :, 0] = (normalized_data[:, :, 0] ).astype(np.float64)
    image[:, :, 1] = (normalized_data[:, :, 1] ).astype(np.float64)

    return image


def IMAGEN_TO_data(image):
    """
    La funcion no es del todo general y considera datos de desplzamiento 2D
    out: los Datos salen normalizados 
    """

    image_in = image.copy()
    normalized_data_out = image_in[:,:,:2] #/ 255.0
    normalized_data_out = normalized_data_out.flatten()


    return normalized_data_out