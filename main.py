import os
from data_loader import load_data, balance_data, create_generators
from models import create_efficientnet
from train import train_model, plot_training_history
from evaluation import evaluate_model, average_fusion
from visualization import plot_class_collage, show_image_samples, show_image_samples_with_predictions

def main():
    # Configuration
    data_dir = '/kaggle/input/monkeypox-skin-lesion-dataset/Original Images/Original Images'
    working_dir = './'
    img_size = (224, 224)
    batch_size = 20
    epochs = 100
    ask_epoch = 5
    class_names = ['Monkey Pox', 'Others']
    
    # Load and prepare data
    train_df, valid_df, test_df = load_data(data_dir)
    train_df = balance_data(train_df, n=200, working_dir=working_dir, img_size=img_size)
    
    # Create data generators
    train_gen, valid_gen, test_gen, test_steps = create_generators(
        train_df, valid_df, test_df, img_size, batch_size
    )
    
    # Visualize data samples
    plot_class_collage(train_df, class_names)
    show_image_samples(train_gen)
    
    # Create and train model
    model = create_efficientnet(img_size + (3,), len(class_names))
    history = train_model(model, train_gen, valid_gen, epochs, ask_epoch)
    plot_training_history(history)
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_gen)
    show_image_samples_with_predictions(test_gen, model)
    
    # Save model
    model.save('monkeypox_model.keras')
    

if __name__ == "__main__":
    main()